import numpy as np
from safe_control.position_control.backup_cbf_qp import BackupCBF
from safe_control.shielding.mps import MPS
from safe_control.shielding.gatekeeper import Gatekeeper

class SafetyFilterWrapper:
    def __init__(self):
        pass
    def get_safe_control(self, state, u_nominal):
        pass
    def reset(self):
        pass

class BackupCBFWrapper(SafetyFilterWrapper):
    def __init__(self, robot, robot_spec, backup_controller, dt=0.05, backup_horizon=2.0):
        self.cbf = BackupCBF(robot, robot_spec, dt=dt, backup_horizon=backup_horizon)
        self.cbf.set_backup_controller(backup_controller)
        # Use a lambda to return the nominal input we pass in get_safe_control
        self._u_nom = np.zeros((2, 1))
        self.cbf.set_nominal_controller(lambda x: self._u_nom)

    def get_safe_control(self, state, u_nominal):
        self._u_nom = u_nominal.reshape(-1, 1)
        # solve_control_problem returns control input
        u_safe = self.cbf.solve_control_problem(state)
        return u_safe

    def is_active(self, state, u_nominal):
        self.get_safe_control(state, u_nominal)
        return self.cbf.is_using_backup()

    def reset(self):
        # BackupCBF doesn't have much state, but we can set trajectories to None if they existed
        pass

class MPSWrapper(SafetyFilterWrapper):
    def __init__(self, robot, robot_spec, backup_controller, dt=0.05, backup_horizon=2.0, horizon_discount=None):
        self.mps = MPS(robot, robot_spec, dt=dt, backup_horizon=backup_horizon, safety_margin=0.0)
        if horizon_discount is not None:
            self.mps.horizon_discount = horizon_discount
        self.mps.set_backup_controller(backup_controller)
        self._u_nom = np.zeros((2, 1))
        self.mps.set_nominal_controller(lambda x: self._u_nom)

    def get_safe_control(self, state, u_nominal):
        self._u_nom = u_nominal.reshape(-1, 1)
        # solve_control_problem might return committed control, need to be careful with indexing
        self.mps.committed_x_traj = None 
        self.mps.committed_u_traj = None
        u_safe = self.mps.solve_control_problem(state)
        return u_safe

    def is_active(self, state, u_nominal):
        self.get_safe_control(state, u_nominal)
        return self.mps.is_using_backup()

    def reset(self):
        self.mps.committed_x_traj = None
        self.mps.committed_u_traj = None

class GatekeeperWrapper(SafetyFilterWrapper):
    def __init__(self, robot, robot_spec, backup_controller, dt=0.05, backup_horizon=2.0, horizon_discount=None):
        self.gk = Gatekeeper(robot, robot_spec, dt=dt, backup_horizon=backup_horizon, 
                             safety_margin=0.0, horizon_discount=horizon_discount)
        self.gk.set_backup_controller(backup_controller)
        self._u_nom = np.zeros((2, 1))
        self.gk.set_nominal_controller(lambda x: self._u_nom)

    def get_safe_control(self, state, u_nominal):
        self._u_nom = u_nominal.reshape(-1, 1)
        self.gk.committed_x_traj = None
        self.gk.committed_u_traj = None
        u_safe = self.gk.solve_control_problem(state)
        return u_safe

    def is_active(self, state, u_nominal):
        self.get_safe_control(state, u_nominal)
        return self.gk.is_using_backup()

    def reset(self):
        self.gk.committed_x_traj = None
        self.gk.committed_u_traj = None
