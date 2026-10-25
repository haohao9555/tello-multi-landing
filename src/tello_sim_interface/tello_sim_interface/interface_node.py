"""Interface node."""
from collections import deque

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

LANDED, TAKING_OFF, FLYING, LANDING = 'LANDED', 'TAKING_OFF', 'FLYING', 'LANDING'


def clamp(x, limit=1.0):
    return max(-limit, min(limit, x))


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class TelloSimInterface(Node):
    def __init__(self):
        super().__init__('tello_sim_interface')
        param = self.declare_parameter
        self.v_max_xy = param('v_max_xy', 1.0).value
        self.v_max_z = param('v_max_z', 1.0).value
        self.yaw_rate_max = param('yaw_rate_max', 1.0).value
        self.delay_steps = param('delay_steps', 0).value
        self.takeoff_height = param('takeoff_height', 1.0).value
        self.takeoff_speed = param('takeoff_speed', 0.5).value
        self.land_speed = param('land_speed', 0.4).value
        self.cmd_timeout = param('cmd_timeout', 0.5).value

        self.state = LANDED
        self.odom_received = False
        self.z = 0.0
        self.vz = 0.0
        self.target_z = 0.0
        self.landing_start = None
        self.stalled_since = None
        self.delay_queue = deque()
        self.last_cmd_time = None

        self.sim_cmd_pub = self.create_publisher(Twist, '/tello/sim/cmd_vel', 10)
        self.enable_pub = self.create_publisher(Bool, '/tello/sim/enable', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, '/tello/status', latched)

        self.create_subscription(Twist, '/tello/cmd_vel', self.on_cmd, 10)
        self.create_subscription(Odometry, '/tello/odom', self.on_odom, 10)
        self.create_service(Trigger, '/tello/takeoff', self.on_takeoff)
        self.create_service(Trigger, '/tello/land', self.on_land)

        self.create_timer(0.1, self.on_watchdog)

        self.publish_status()
        self.get_logger().info('Tello interface ready: /tello/cmd_vel, /tello/takeoff, /tello/land')



    def set_state(self, state):
        if state != self.state:
            self.get_logger().info(f'{self.state} -> {state}')
            self.state = state
            self.delay_queue.clear()
            self.publish_status()

    def publish_status(self):
        self.status_pub.publish(String(data=self.state))

    def send_enable(self, on):
        self.enable_pub.publish(Bool(data=on))

    def send_velocity(self, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = vx, vy, vz
        msg.angular.z = yaw_rate
        self.sim_cmd_pub.publish(msg)



    def on_cmd(self, msg):
        self.last_cmd_time = self.get_clock().now()
        if self.state != FLYING:
            return
        out = Twist()
        out.linear.x = clamp(msg.linear.x) * self.v_max_xy
        out.linear.y = clamp(msg.linear.y) * self.v_max_xy
        out.linear.z = clamp(msg.linear.z) * self.v_max_z
        out.angular.z = clamp(msg.angular.z) * self.yaw_rate_max

        self.delay_queue.append(out)
        if len(self.delay_queue) > self.delay_steps:
            self.sim_cmd_pub.publish(self.delay_queue.popleft())

    def on_odom(self, msg):
        self.odom_received = True
        self.z = msg.pose.pose.position.z
        self.vz = msg.twist.twist.linear.z
        now = stamp_to_sec(msg.header.stamp)

        if self.state == TAKING_OFF:
            error = self.target_z - self.z
            if error < 0.03:
                self.send_velocity()
                self.set_state(FLYING)
            else:

                self.send_enable(True)
                self.send_velocity(vz=min(self.takeoff_speed, 1.5 * error))

        elif self.state == LANDING:
            self.send_velocity(vz=-self.land_speed)
            if self.landing_start is None:
                self.landing_start = now

            if now - self.landing_start > 1.0 and self.vz > -0.05:
                if self.stalled_since is None:
                    self.stalled_since = now
                elif now - self.stalled_since > 0.3:
                    self.send_velocity()
                    self.send_enable(False)
                    self.set_state(LANDED)
            else:
                self.stalled_since = None

    def on_watchdog(self):
        self.publish_status()

        if self.state == FLYING and self.last_cmd_time is not None:
            idle = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9
            if idle > self.cmd_timeout:
                self.delay_queue.clear()
                self.send_velocity()

    def on_takeoff(self, request, response):
        if not self.odom_received:
            response.success, response.message = False, 'No /tello/odom received yet; is the simulation running?'
        elif self.state != LANDED:
            response.success, response.message = False, f'Cannot take off from state {self.state}'
        else:
            self.target_z = self.z + self.takeoff_height
            self.send_enable(True)
            self.set_state(TAKING_OFF)
            response.success, response.message = True, f'Taking off to target height {self.target_z:.2f} m'
        return response

    def on_land(self, request, response):
        if self.state in (FLYING, TAKING_OFF):
            self.landing_start = None
            self.stalled_since = None
            self.set_state(LANDING)
            response.success, response.message = True, 'Landing'
        else:
            response.success, response.message = False, f'Cannot land from state {self.state}'
        return response


def main():
    rclpy.init()
    node = TelloSimInterface()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
