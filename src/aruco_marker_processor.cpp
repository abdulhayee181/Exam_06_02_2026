#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <set>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>

class ArucoMarkerProcessor : public rclcpp::Node
{
public:
    ArucoMarkerProcessor() : Node("aruco_marker_processing")
    {
        // Parameters
        camera_topic = "/camera/image";
        output_topic = "/processed_image";
        linear_scan_speed = 0.2;  // Speed for moving along the line
        scan_distance_limit = 10.0; // Distance to travel to find all markers
        
        // Subscriptions
        image_sub = this->create_subscription<sensor_msgs::msg::Image>(
            camera_topic, 10,
            std::bind(&ArucoMarkerProcessor::image_callback, this, std::placeholders::_1));
            
        odom_sub = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            std::bind(&ArucoMarkerProcessor::odom_callback, this, std::placeholders::_1));
        
        // Publishers
        cmd_vel_pub = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        image_pub = this->create_publisher<sensor_msgs::msg::Image>(output_topic, 10);
        
        // Initialize ArUco detector
        dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL);
        parameters = cv::aruco::DetectorParameters::create();
        
        // State variables
        state = SCANNING;
        current_x = 0.0;
        start_x = 0.0;
        first_odom_received = false;
        current_target_index = 0;
        is_centered = false;
        
        // Visual servoing parameters
        kp_angular = 0.01;
        kp_linear = 0.5;
        center_threshold = 10.0;
        target_distance_pixels = 200.0; // Target size in pixels to maintain distance
        wait_duration = 5.0;
        
        RCLCPP_INFO(this->get_logger(), "=== Linear ArUco Marker Processor Node ===");
        RCLCPP_INFO(this->get_logger(), "State: SCANNING (Moving along the line)");
    }

private:
    enum State { SCANNING, SORTING, SEARCHING, CENTERING, COMPLETE };
    
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        current_x = msg->pose.pose.position.x;
        if (!first_odom_received) {
            start_x = current_x;
            first_odom_received = true;
        }
    }
    
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
            cv::Mat image = cv_ptr->image;
            cv::Mat display_image = image.clone();
            
            std::vector<int> marker_ids;
            std::vector<std::vector<cv::Point2f>> marker_corners;
            cv::aruco::detectMarkers(image, dictionary, marker_corners, marker_ids, parameters);
            
            switch (state) {
                case SCANNING:
                    handle_scanning_state(marker_ids, marker_corners, display_image);
                    break;
                case SORTING:
                    handle_sorting_state(display_image);
                    break;
                case SEARCHING:
                    handle_searching_state(marker_ids, marker_corners, display_image);
                    break;
                case CENTERING:
                    handle_centering_state(marker_ids, marker_corners, display_image, msg->header);
                    break;
                case COMPLETE:
                    handle_complete_state(display_image, msg->header);
                    break;
            }
            
            cv::imshow("Linear Marker Processor", display_image);
            cv::waitKey(1);
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception: %s", e.what());
        }
    }
    
    void handle_scanning_state(const std::vector<int>& marker_ids,
                              const std::vector<std::vector<cv::Point2f>>& marker_corners,
                              cv::Mat& display_image)
    {
        // Discover markers and store approximate locations
        for (int id : marker_ids) {
            if (marker_locations.find(id) == marker_locations.end()) {
                marker_locations[id] = current_x;
                RCLCPP_INFO(this->get_logger(), "Discovered marker ID %d at approx x=%.2f", id, current_x);
            }
        }
        
        if (!marker_ids.empty()) {
            cv::aruco::drawDetectedMarkers(display_image, marker_corners, marker_ids);
        }
        
        // Move forward along the line
        auto twist_msg = geometry_msgs::msg::Twist();
        twist_msg.linear.x = linear_scan_speed;
        cmd_vel_pub->publish(twist_msg);
        
        // End scan after set distance
        if (std::abs(current_x - start_x) >= scan_distance_limit) {
            state = SORTING;
            cmd_vel_pub->publish(geometry_msgs::msg::Twist()); // Stop
            RCLCPP_INFO(this->get_logger(), "=== LINEAR SCAN COMPLETE ===");
        }
        
        cv::putText(display_image, "State: Linear Scanning", cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    }
    
    void handle_sorting_state(cv::Mat& display_image)
    {
        sorted_marker_ids.clear();
        for (auto const& [id, loc] : marker_locations) {
            sorted_marker_ids.push_back(id);
        }
        std::sort(sorted_marker_ids.begin(), sorted_marker_ids.end());
        
        if (!sorted_marker_ids.empty()) {
            current_target_index = 0;
            state = SEARCHING;
            RCLCPP_INFO(this->get_logger(), "Visiting Marker ID: %d", sorted_marker_ids[current_target_index]);
        } else {
            state = COMPLETE;
        }
    }
    
    void handle_searching_state(const std::vector<int>& marker_ids,
                               const std::vector<std::vector<cv::Point2f>>& marker_corners,
                               cv::Mat& display_image)
    {
        int target_id = sorted_marker_ids[current_target_index];
        auto it = std::find(marker_ids.begin(), marker_ids.end(), target_id);
        
        if (it != marker_ids.end()) {
            state = CENTERING;
            is_centered = false;
            center_start_time = this->now();
        } else {
            // Move toward the recorded location of the marker
            double target_x = marker_locations[target_id];
            auto twist_msg = geometry_msgs::msg::Twist();
            twist_msg.linear.x = (target_x > current_x) ? 0.2 : -0.2;
            // Add a small rotation to keep looking toward the markers
            twist_msg.angular.z = 0.1; 
            cmd_vel_pub->publish(twist_msg);
        }
    }
    
    void handle_centering_state(const std::vector<int>& marker_ids,
                            const std::vector<std::vector<cv::Point2f>>& marker_corners,
                            cv::Mat& display_image,
                            const std_msgs::msg::Header& header)
    {
        int target_id = sorted_marker_ids[current_target_index];
        auto it = std::find(marker_ids.begin(), marker_ids.end(), target_id);
        
        if (it != marker_ids.end()) {
            size_t idx = std::distance(marker_ids.begin(), it);
            cv::Point2f center(0, 0);
            for (const auto& corner : marker_corners[idx]) {
                center.x += corner.x;
                center.y += corner.y;
            }
            center.x /= 4.0; center.y /= 4.0;
            
            double error_x = center.x - (display_image.cols / 2.0);
            
            if (!is_centered) {
                if (std::abs(error_x) > center_threshold) {
                    auto twist_msg = geometry_msgs::msg::Twist();
                    twist_msg.angular.z = -kp_angular * error_x;
                    cmd_vel_pub->publish(twist_msg);
                } else {
                    is_centered = true;
                    center_start_time = this->now();
                    cmd_vel_pub->publish(geometry_msgs::msg::Twist());
                }
            } else {
                double elapsed = (this->now() - center_start_time).seconds();
                if (elapsed >= wait_duration) {
                    current_target_index++;
                    state = (current_target_index < sorted_marker_ids.size()) ? SEARCHING : COMPLETE;
                }
            }
            
            cv::circle(display_image, center, 25, cv::Scalar(0, 255, 0), 3);
            publish_processed_image(display_image, header);
        } else {
            state = SEARCHING; // Lost it, go back to finding
        }
    }
    
    void handle_complete_state(cv::Mat& display_image, const std_msgs::msg::Header& header)
    {
        cv::putText(display_image, "LINEAR MISSION COMPLETE!", cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cmd_vel_pub->publish(geometry_msgs::msg::Twist());
        publish_processed_image(display_image, header);
    }
    
    void publish_processed_image(const cv::Mat& image, const std_msgs::msg::Header& header)
    {
        auto msg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
        image_pub->publish(*msg);
    }

    // Variables
    std::string camera_topic, output_topic;
    double linear_scan_speed, scan_distance_limit, current_x, start_x;
    double kp_angular, kp_linear, center_threshold, target_distance_pixels, wait_duration;
    bool first_odom_received, is_centered;
    
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub;
    
    cv::Ptr<cv::aruco::Dictionary> dictionary;
    cv::Ptr<cv::aruco::DetectorParameters> parameters;
    
    State state;
    std::map<int, double> marker_locations; // ID to X-coordinate
    std::vector<int> sorted_marker_ids;
    size_t current_target_index;
    rclcpp::Time center_start_time;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArucoMarkerProcessor>());
    rclcpp::shutdown();
    return 0;
}
