from system_manager_msg.action import ManipulationTask
from system_manager_msg.srv import TeleopFrame
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse

class EHIClient():
    def __init__(self, node):
        self.node = node

        self.action_server_ = ActionServer(
            node=self.node,
            action_type=ManipulationTask,
            action_name='/system_manager/manipulation/action',
            execute_callback=self.execute_callback,
            goal_callback=self.handle_goal,
            cancel_callback=self.handle_cancel,
            handle_accepted_callback=self.handle_accepted
        )

        self.teleop_server = self.node.create_service(
            TeleopFrame,
            "/system_manager/teleop/service",
            self._teleop_callback,
        )

        self.manip_type = 0
        self.manip_object= 0 
        self.manip_action = 0
        self.object_bbox = None
        self.prompt = ""

        self.is_stop = True
        self.model_in_control = True

    async def execute_callback(self, goal_handle):
        feedback_msg = ManipulationTask.Feedback()
        feedback_msg.progress = 0
        goal_handle.publish_feedback(feedback_msg)

    def handle_goal(self, goal_handle):
        return GoalResponse.ACCEPT

    def handle_cancel(self, goal_handle):
        self.is_stop = True
        return CancelResponse.ACCEPT

    def handle_accepted(self, goal_handle):
        goal = goal_handle.request
        self.manip_type = goal.manip_type
        self.manip_object = goal.manip_object
        self.manip_action = goal.manip_action
        self.bbox = goal.object_bbox
        self.prompt = goal.reserved
        self.is_stop = False

        print(f"Manip Type: {self.manip_type}")
        print(f"Manip Object: {self.manip_object}")
        print(f"Manip Action: {self.manip_action}")
        print(f"Bbox: {self.bbox}")
        print(f"Prompt: {self.prompt}")

    def _teleop_callback(self, request, response):
        print(f'recv teleop request: {request.action}')
        if request.action == 0:
            print(f'Human in control')
            self.model_in_control = False
        elif request.action == 1:
            print(f'Model in control')
            self.model_in_control = True
        else:
            print(f'Unsupport action: {request.action}')

    def stop(self):
        return self.is_stop

    def run(self):
        while rclpy.ok():
            rclpy.spin(self.node)
            rclpy.sleep(0.01)

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("EHITestClient")
    ehi_client = EHIClient(node)
    ehi_client.run()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
