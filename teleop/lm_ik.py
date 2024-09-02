import mujoco as mj
import numpy as np

class LevenbegMarquardtIK:
    
    def __init__(self, model, data, step_size=0.5, tol=0.01, damping=0.005):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.damping = damping
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.term_variable_tol = 0.0001
    
    def check_joint_limits(self, q):
        """Check if the joints is under or above its limits"""
        for i in range(len(q)):
            if i in [6, 7, 14, 15]:
                continue
            q[i] = max(self.model.jnt_range[i][0], 
                       min(q[i], self.model.jnt_range[i][1]))

    #Levenberg-Marquardt pseudocode implementation
    def calculate(self, xpos_goals, quat_goals, init_q, body_ids, max_iter=200):
        mj.mj_resetData(self.model, self.data)

        """Calculate the desire joints angles for goal"""
        if not isinstance(xpos_goals, list):
            xpos_goals = [xpos_goals]
        if not isinstance(quat_goals, list):
            quat_goals = [quat_goals]
        if not isinstance(body_ids, list):
            body_ids = [body_ids]

        self.data.qpos = init_q.copy()
        mj.mj_forward(self.model, self.data)
        current_poses = \
            np.vstack([self.data.body(body_id).xpos for body_id in body_ids])
        current_quats = \
            np.vstack([self.data.body(body_id).xquat for body_id in body_ids])
        quat_errors = np.zeros((len(body_ids), 3))
        for i in range(len(body_ids)):
            current_quat_inv, quat_error_ = np.zeros(4), np.zeros(4)
            mj.mju_negQuat(current_quat_inv, current_quats[i])
            quat_error_ = np.zeros(4)
            mj.mju_mulQuat(quat_error_, quat_goals[i], current_quat_inv)
            mj.mju_quat2Vel(quat_errors[i], quat_error_, 1)

        xpos_errors = xpos_goals - current_poses
        total_error = np.sum(np.linalg.norm(xpos_errors, axis=0)) + \
            np.sum(np.linalg.norm(quat_errors, axis=0))

        # print("----")
        # print("initial position error: {:.3}".format(np.sum(np.linalg.norm(xpos_errors, axis=0))))
        # print("initial rotation error: {:.3}".format(np.sum(np.linalg.norm(quat_errors, axis=0))))
        num_iter = 0
        while (total_error >= self.tol) and num_iter < max_iter:
            num_iter += 1
            delta_q = np.zeros(len(init_q))
            for i, body_id in enumerate(body_ids):
                mj.mj_jacBody(self.model, self.data, self.jacp, self.jacr, body_id)
                n = self.jacp.shape[1]
                I = np.identity(n)
                product = self.jacp.T @ self.jacp + self.damping * I
                if np.isclose(np.linalg.det(product), 0):
                    j_inv = np.linalg.pinv(product) @ self.jacp.T
                else:
                    j_inv = np.linalg.inv(product) @ self.jacp.T

                delta_q += j_inv @ xpos_errors[i]

                n = self.jacr.shape[1]
                I = np.identity(n)
                product = self.jacr.T @ self.jacr + self.damping * I
            
                if np.isclose(np.linalg.det(product), 0):
                    j_inv = np.linalg.pinv(product) @ self.jacr.T
                else:
                    j_inv = np.linalg.inv(product) @ self.jacr.T

                delta_q += j_inv @ quat_errors[i]

            # terminate if qpos do not change
            new_qpos = self.data.qpos + self.step_size * delta_q
            if np.linalg.norm(new_qpos - self.data.qpos) < self.term_variable_tol:
                break
            self.data.qpos = new_qpos

            # check limits
            self.check_joint_limits(self.data.qpos)

            # compute forward kinematics
            mj.mj_forward(self.model, self.data)
            
            # calculate new error
            current_poses = np.vstack([self.data.xpos[body_id] for body_id in body_ids])
            current_quats = np.vstack([self.data.xquat[body_id] for body_id in body_ids])
            quat_errors = np.zeros((len(body_ids), 3))
            for i in range(len(body_ids)):
                current_quat_inv, quat_error_ = np.zeros(4), np.zeros(4)
                mj.mju_negQuat(current_quat_inv, current_quats[i])
                quat_error_ = np.zeros(4)
                mj.mju_mulQuat(quat_error_, quat_goals[i], current_quat_inv)
                mj.mju_quat2Vel(quat_errors[i], quat_error_, 1)
            xpos_errors = xpos_goals - current_poses
            total_error = np.sum(np.linalg.norm(xpos_errors, axis=0)) + \
                np.sum(np.linalg.norm(quat_errors, axis=0))

            # if num_iter % 100 == 0:
            #     print("position error {} at iter {}".format(
            #         np.sum(np.linalg.norm(xpos_errors, axis=0)), num_iter))
            #     print("rotation error {} at iter {}".format(np.sum(
            #         np.linalg.norm(quat_errors, axis=0)), num_iter))
        
        # print("position error {:.3} at iter {}".format(
        #     np.sum(np.linalg.norm(xpos_errors, axis=0)), num_iter))
        # print("rotation error {:.3} at iter {}".format(
        #     np.sum(np.linalg.norm(quat_errors, axis=0)), num_iter))
        # print("----")

        self.check_joint_limits(self.data.qpos)
        return self.data.qpos.copy()
