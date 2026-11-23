"""Authored connection projection on independent rigid transforms, not bones."""
import numpy as np
from mathutils import Matrix, Euler, Vector


def project_authored_joint(predicted, parent_delta, parameters):
    def rigid(value):
        a=np.asarray(value,dtype=float)
        if a.shape!=(4,4) or not np.isfinite(a).all() or not np.allclose(a[3],[0,0,0,1],atol=1e-7):raise ValueError('连接矩阵非法')
        if not np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),atol=2e-6) or np.linalg.det(a[:3,:3])<0:raise ValueError('连接不支持缩放或镜像')
        return Matrix(a.tolist())
    t=rigid(predicted);parent=rigid(parent_delta)
    j=Euler(parameters['rotation_rad'],parameters['rotation_order']).to_matrix().to_4x4()
    j.translation=Vector(parameters['position_m']);anchor=parent@j
    relative=anchor.inverted()@t@j
    angles=relative.to_3x3().to_euler(parameters['rotation_order'])
    for i in range(3):angles[i]=min(parameters['angular_max_rad'][i],max(parameters['angular_min_rad'][i],angles[i]))
    allowed=angles.to_matrix().to_4x4()
    allowed.translation=Vector([min(parameters['linear_max_m'][i],max(parameters['linear_min_m'][i],relative.translation[i])) for i in range(3)])
    result=anchor@allowed@j.inverted()
    if not np.isfinite(np.array(result)).all():raise ValueError('连接投影产生非法结果')
    return result
