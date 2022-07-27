from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pyglet


class ArmEnv(object):
    viewer = None
    viewer1 = None
    dt = .1    # refresh rate
    action_bound = [-1, 1]
    goal3D = {'x': 100., 'y': 100., 'z': 100., 'l': 40}
    goal = {'x': 100., 'y': 100., 'l': 40}
    obstacle1 = {'x': -50., 'y': 0., 'z': 80., 'l':32}
    obstacle = {'x': 150., 'y': 280., 'l': 32}
    state_dim = 18
    action_dim = 3
    fig = plt.figure()
    # ax = plt.axes(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot(1, 1, 1, projection='3d')  # 为了画出连续的效果
    bias = 10

    def __init__(self):
        self.arm_info = np.zeros(
            3, dtype=[('l', np.float32), ('r', np.float32)])
        self.arm_info['l'] = 70        # 3 arms length
        self.arm_info['r'] = np.pi/6    # 3 angles information

        self.arm_info1 = np.zeros(
            1, dtype=[('l', np.float32), ('r', np.float32)])
        self.arm_info1['l'] = 70  # 1 arm length
        self.arm_info1['r'] = np.pi / 6  # 1 angle information

        self.a2l = 0
        self.a1l = 0
        self.a3z = 0
        self.a2z = 0
        self.a1z = 0

        self.on_goal = 0

        self.KP = 2
        self.KI = 0
        self.KD = 1
        self.u = 0
        self.e_all = 0
        self.e_last = 0


        self.on_touch = 0

    def step(self, action):

        #### move the goal

        if self.goal3D['x'] < 120:
            self.goal3D['x'] += 5

        else:
            self.goal3D['x'] = -120

        # 三维转二维
        lenth = np.sqrt(np.square(self.goal3D['x']) + np.square(self.goal3D['y']))
        self.goal['x'] = 200 - lenth
        self.goal['y'] = self.goal3D['z'] + 200

        #### 预先三维转化二维
        if self.goal3D['x'] == 0:
            if self.goal3D['y'] >= 0:
                self.f = np.pi / 2 + np.arctan(self.bias / lenth)
            else:
                self.f = 3 * np.pi / 2+ np.arctan(self.bias / lenth)
        else:
            self.f = np.arctan(self.goal3D['y'] / self.goal3D['x'])+ np.arctan(self.bias / lenth)




        #### PID 控制

        # 更新夹角位置（获得输出）
        self.arm_info1['r'][0] += self.u*self.dt
        self.arm_info1['r'][0] %= np.pi * 2  # normalize

        # 计算偏差 deata
        if self.goal3D['x'] >= 0:
            deata = self.f % (2 * np.pi) - self.arm_info1['r'][0]
        else:
            deata = self.f + np.pi - self.arm_info1['r'][0]

        # pid 控制
        self.u = self.KP*deata + self.KI*self.e_all + self.KD*(deata-self.e_last)
        self.e_all += deata # 误差的累加和
        self.e_last = deata # 前一个误差值


        #### DDPG 控制

        done = False
        action = np.clip(action, *self.action_bound)
        self.arm_info['r'] += action * self.dt
        self.arm_info['r'] %= np.pi * 2    # normalize

        # 计算机械臂平面位置
        (a1l, a2l, a3l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r, a3r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([200., 200.])    # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        a2xy = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        finger = np.array([np.cos(a1r + a2r + a3r), np.sin(a1r + a2r + a3r)]) * a3l + a2xy  # a3 end (x3, y3)

        self.arm_info1['l'][0] = finger[0]-200
        self.a2l = a2xy[0] - 200
        self.a1l = a1xy_[0] - 200

        self.a3z = finger[1]-200
        self.a2z = a2xy[1] - 200
        self.a1z = a1xy_[1] - 200

        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0]) / 400, (self.goal['y'] - a1xy_[1]) / 400]
        dist2 = [(self.goal['x'] - a2xy[0]) / 400, (self.goal['y'] - a2xy[1]) / 400]
        dist3 = [(self.goal['x'] - finger[0]) / 400, (self.goal['y'] - finger[1]) / 400]

        dist4 = [(self.obstacle['x'] - a1xy_[0]) / 400, (self.obstacle['y'] - a1xy_[1]) / 400]
        dist5 = [(self.obstacle['x'] - a2xy[0]) / 400, (self.obstacle['y'] - a2xy[1]) / 400]

        # 构建奖惩函数
        r = -np.sqrt(dist3[0] ** 2 + dist3[1] ** 2) + 0.2*(np.sqrt(dist4[0]**2+dist4[1]**2)+np.sqrt(dist5[0]**2+dist5[1]**2))



        # add obstacle
        f1 = (a2xy-finger)/5+finger
        f2 = 2*(a2xy-finger)/5+finger
        f3 = 3*(a2xy-finger)/5+finger
        f4 = 4*(a2xy-finger)/5+finger

        a1 = (a1xy-a1xy_)/5+a1xy_
        a2 = 2*(a1xy-a1xy_)/5+a1xy_
        a3 = 3*(a1xy-a1xy_)/5+a1xy_
        a4 = 4*(a1xy-a1xy_)/5+a1xy_

        e1 = (a2xy - a1xy_) / 5 + a1xy_
        e2 = 2 * (a2xy - a1xy_) / 5 + a1xy_
        e3 = 3 * (a2xy - a1xy_) / 5 + a1xy_
        e4 = 4 * (a2xy - a1xy_) / 5 + a1xy_

        # if self.obstacle['x'] - self.obstacle['l']/2 < f1[0] or f2[0] or f3[0] or a1[0] or a2[0] or a3[0] or a1xy_[0] < self.obstacle['x'] + self.obstacle['l']:
        #     if self.obstacle['y'] - self.obstacle['l']/2 < f1[1] or f2[1] or f3[1] or a1[1] or a2[1] or a3[1] or a1xy_[1] < self.obstacle['y'] + self.obstacle['l']:
        #         r += -5
        #         self.on_touch = 1

        R = 2.5
        if self.obstacle['x'] - self.obstacle['l']/2 < f1[0] < self.obstacle['x'] + self.obstacle['l'] and \
                                        self.obstacle['y'] - self.obstacle['l']/2 < f1[1] < self.obstacle['y'] + self.obstacle['l']:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l']/2 < f2[0] < self.obstacle['x'] + self.obstacle['l'] and \
                                        self.obstacle['y'] - self.obstacle['l']/2 < f2[1] < self.obstacle['y'] + self.obstacle['l']:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l']/2 < f3[0] < self.obstacle['x'] + self.obstacle['l'] and \
                                        self.obstacle['y'] - self.obstacle['l']/2 < f3[1] < self.obstacle['y'] + self.obstacle['l']:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l']/2 < f4[0] < self.obstacle['x'] + self.obstacle['l'] and \
                                        self.obstacle['y'] - self.obstacle['l']/2 < f4[1] < self.obstacle['y'] + self.obstacle['l']:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l']/2 < a1[0] < self.obstacle['x'] + self.obstacle['l'] and \
                                        self.obstacle['y'] - self.obstacle['l']/2 < a1[1] < self.obstacle['y'] + self.obstacle['l']:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l']/2 < a2[0] < self.obstacle['x'] + self.obstacle['l'] and \
                                        self.obstacle['y'] - self.obstacle['l']/2 < a2[1] < self.obstacle['y'] + self.obstacle['l']:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l']/2 < a3[0] < self.obstacle['x'] + self.obstacle['l'] and \
                                        self.obstacle['y'] - self.obstacle['l']/2 < a3[1] < self.obstacle['y'] + self.obstacle['l']:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l']/2 < a4[0] < self.obstacle['x'] + self.obstacle['l'] and \
                                        self.obstacle['y'] - self.obstacle['l']/2 < a4[1] < self.obstacle['y'] + self.obstacle['l']:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l']/2 < e1[0] < self.obstacle['x'] + self.obstacle['l'] and \
                                        self.obstacle['y'] - self.obstacle['l']/2 < e1[1] < self.obstacle['y'] + self.obstacle['l']:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l']/2 < e2[0] < self.obstacle['x'] + self.obstacle['l'] and \
                                        self.obstacle['y'] - self.obstacle['l']/2 < e2[1] < self.obstacle['y'] + self.obstacle['l']:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l']/2 < e3[0] < self.obstacle['x'] + self.obstacle['l'] and \
                                        self.obstacle['y'] - self.obstacle['l']/2 < e3[1] < self.obstacle['y'] + self.obstacle['l']:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l']/2 < e4[0] < self.obstacle['x'] + self.obstacle['l'] and \
                                        self.obstacle['y'] - self.obstacle['l']/2 < e4[1] < self.obstacle['y'] + self.obstacle['l']:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l']/2 < a2xy[0] < self.obstacle['x'] + self.obstacle['l'] and \
                                        self.obstacle['y'] - self.obstacle['l']/2 < a2xy[1] < self.obstacle['y'] + self.obstacle['l']:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l']/2 < a1xy_[0] < self.obstacle['x'] + self.obstacle['l'] and \
                                        self.obstacle['y'] - self.obstacle['l']/2 < a1xy_[1] < self.obstacle['y'] + self.obstacle['l']:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l']/2 < finger[0] < self.obstacle['x'] + self.obstacle['l'] and \
                                        self.obstacle['y'] - self.obstacle['l']/2 < finger[1] < self.obstacle['y'] + self.obstacle['l']:
            r += -R
            self.on_touch = 1

        # done and reward
        if self.goal['x'] - self.goal['l']/2 < finger[0] < self.goal['x'] + self.goal['l']/2:
            if self.goal['y'] - self.goal['l']/2 < finger[1] < self.goal['y'] + self.goal['l']/2:
                r += 1.
                self.on_goal += 1
                if self.on_goal > 50:
                    done = True
        else:
            self.on_goal = 0

        # state
        s = np.concatenate((a1xy_ / 200, a2xy / 200, finger / 200, dist1 + dist2 + dist3, [1. if self.on_goal else 0.],
                            dist4 + dist5, [0. if self.on_touch else 1.]))
        return s, r, done

    def reset(self):

        self.ax.scatter3D(0,0,0, cmap='Blues')
        plt.ion()
        plt.show()

        self.goal3D['x'] = (np.random.rand()-0.5)*280.
        self.goal3D['y'] = (np.random.rand()-0.5)*280.
        self.goal3D['z'] = (np.random.rand()-0.5)*280.

        # self.goal3D['x'] = -140
        # self.goal3D['y'] = -140
        # self.goal3D['z'] = -140

        # print(self.goal3D['x'],self.goal3D['y'],self.goal3D['z'])
        self.arm_info['r'] = 2 * np.pi * np.random.rand(3)
        self.arm_info1['r'] = 2 * np.pi * np.random.rand(1)
        self.on_goal = 0

        self.u = 0
        self.e_all = 0
        self.e_last = 0

        # 三维转二维
        lenth = np.sqrt(np.square(self.goal3D['x']) + np.square(self.goal3D['y']))
        self.goal['x'] = 200 - lenth
        self.goal['y'] = self.goal3D['z'] + 200

        # 计算goal的矢量夹角
        if self.goal3D['x'] == 0:
            if self.goal3D['y'] >= 0:
                self.f = np.pi/2+ np.arctan(self.bias / lenth)
            else:
                self.f = 3*np.pi/2+ np.arctan(self.bias / lenth)
        else:
            self.f = np.arctan(self.goal3D['y']/self.goal3D['x'])+ np.arctan(self.bias / lenth)




        (a1l, a2l, a3l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r, a3r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([200., 200.])  # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        a2xy = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        finger = np.array([np.cos(a1r + a2r + a3r), np.sin(a1r + a2r + a3r)]) * a3l + a2xy  # a3 end (x3, y3)

        self.arm_info1['l'][0] = finger[0]-200


        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0]) / 400, (self.goal['y'] - a1xy_[1]) / 400]
        dist2 = [(self.goal['x'] - a2xy[0]) / 400, (self.goal['y'] - a2xy[1]) / 400]
        dist3 = [(self.goal['x'] - finger[0]) / 400, (self.goal['y'] - finger[1]) / 400]

        dist4 = [(self.obstacle['x'] - a1xy_[0]) / 400, (self.obstacle['y'] - a1xy_[1]) / 400]
        dist5 = [(self.obstacle['x'] - a2xy[0]) / 400, (self.obstacle['y'] - a2xy[1]) / 400]

        # state
        s = np.concatenate((a1xy_ / 200, a2xy / 200, finger / 200, dist1 + dist2 + dist3, [1. if self.on_goal else 0.],
                            dist4 + dist5, [0. if self.on_touch else 1.]))
        return s

    def render(self):

        a3l = np.fabs(self.arm_info1['l'][0])
        a3r = self.arm_info1['r'][0]
        a3xy = np.array([np.cos(a3r), np.sin(a3r)]) * a3l

        a2l = np.fabs(self.a2l)
        a2xy = np.array([np.cos(a3r), np.sin(a3r)]) * a2l

        a1l = np.fabs(self.a1l)
        a1xy = np.array([np.cos(a3r), np.sin(a3r)]) * a1l

        a0xy = np.array([0,0])

        a3z = [self.a3z]
        a2z = [self.a2z]
        a1z = [self.a1z]
        a0z = [0]

        a0 = np.concatenate((a0xy,a0z))
        a1 = np.concatenate((a1xy,a1z))
        a2 = np.concatenate((a2xy,a2z))
        a3 = np.concatenate((a3xy,a3z))

        A = np.array([a0,a1,a2,a3])

        x = A[:,0]
        y = A[:,1]
        z = A[:,2]

        xa = self.goal3D['x']
        ya = self.goal3D['y']
        za = self.goal3D['z']
        l = self.goal3D['l']/2

        xd = np.array([xa - l, xa + l, xa + l, xa - l, xa - l, xa + l, xa + l, xa - l,
                       xa - l, xa - l, xa - l, xa - l, xa + l, xa + l, xa + l, xa + l])
        yd = np.array([ya - l, ya - l, ya + l, ya + l, ya + l, ya + l, ya - l, ya - l,
                       ya - l, ya + l, ya + l, ya - l, ya - l, ya - l, ya + l, ya + l])
        zd = np.array([za - l, za - l, za - l, za - l, za + l, za + l, za + l, za + l,
                       za - l, za - l, za + l, za + l, za + l, za - l, za - l, za + l])

        xa = self.obstacle1['x']
        ya = self.obstacle1['y']
        za = self.obstacle1['z']
        l = self.obstacle1['l'] / 2

        xdo = np.array([xa - l, xa + l, xa + l, xa - l, xa - l, xa + l, xa + l, xa - l,
                       xa - l, xa - l, xa - l, xa - l, xa + l, xa + l, xa + l, xa + l])
        ydo = np.array([ya - l, ya - l, ya + l, ya + l, ya + l, ya + l, ya - l, ya - l,
                       ya - l, ya + l, ya + l, ya - l, ya - l, ya - l, ya + l, ya + l])
        zdo = np.array([za - l, za - l, za - l, za - l, za + l, za + l, za + l, za + l,
                       za - l, za - l, za + l, za + l, za + l, za - l, za - l, za + l])


        lines = self.ax.plot3D(x,y,z,'red',lw=5)
        lines1 = self.ax.plot3D(xd,yd,zd,'blue',lw=2)
        lines2 = self.ax.plot3D(xdo,ydo,zdo,'blue',lw=1)
        self.ax.set_xlim3d(-150,150)
        self.ax.set_ylim3d(-150,150)
        self.ax.set_zlim3d(-150,150)
        self.ax.set_aspect('equal')
        plt.pause(0.1)
        self.ax.lines.remove(lines[0])
        self.ax.lines.remove(lines1[0])
        self.ax.lines.remove(lines2[0])

        # if self.viewer is None:
        #     self.viewer = Viewer(self.arm_info, self.goal)
        # self.viewer.render()
        # if self.viewer1 is None:
        #     self.viewer1 = Round(self.arm_info1, self.goal3D)
        # self.viewer1.render()

    def sample_action(self):
        return np.random.rand(3)-0.5    # two radians

class Viewer(pyglet.window.Window):
    bar_thc = 5

    def __init__(self, arm_info, goal):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.arm_info = arm_info
        self.goal_info = goal
        # self.obstacle_info = obstacle
        self.center_coord = np.array([200, 200])
        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,                # location
                     goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
            ('c3B', (86, 109, 249) * 4))    # color
        # self.obstacle = self.batch.add(
        #     4, pyglet.gl.GL_QUADS, None,  # 4 corners
        #     ('v2f', [obstacle['x'] - obstacle['l'] / 2, obstacle['y'] - obstacle['l'] / 2,  # location
        #              obstacle['x'] - obstacle['l'] / 2, obstacle['y'] + obstacle['l'] / 2,
        #              obstacle['x'] + obstacle['l'] / 2, obstacle['y'] + obstacle['l'] / 2,
        #              obstacle['x'] + obstacle['l'] / 2, obstacle['y'] - obstacle['l'] / 2]),
        #     ('c3B', (56, 209, 249) * 4))  # color
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,                # location
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))    # color
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,              # location
                     100, 160,
                     200, 160,
                     200, 150]), ('c3B', (249, 86, 86) * 4,))
        self.arm3 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,  # location
                     100, 160,
                     200, 160,
                     200, 150]), ('c3B', (249, 86, 86) * 4,))

    def render(self):
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_arm(self):
        # update goal
        self.goal.vertices = (
            self.goal_info['x'] - self.goal_info['l']/2, self.goal_info['y'] - self.goal_info['l']/2,
            self.goal_info['x'] + self.goal_info['l']/2, self.goal_info['y'] - self.goal_info['l']/2,
            self.goal_info['x'] + self.goal_info['l']/2, self.goal_info['y'] + self.goal_info['l']/2,
            self.goal_info['x'] - self.goal_info['l']/2, self.goal_info['y'] + self.goal_info['l']/2)

        # update arm
        (a1l, a2l, a3l) = self.arm_info['l']     # radius, arm length
        (a1r, a2r, a3r) = self.arm_info['r']     # radian, angle
        a1xy = self.center_coord            # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy   # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * a2l + a1xy_  # a2 end and a3 start (x2, y2)
        a3xy_ = np.array([np.cos(a1r + a2r + a3r), np.sin(a1r + a2r + a3r)]) * a3l + a2xy_  # a3 end (x3, y3)

        a1tr, a2tr  = np.pi / 2 - self.arm_info['r'][0], np.pi / 2 - self.arm_info['r'][0] - self.arm_info['r'][1]
        a3tr = np.pi / 2 - self.arm_info['r'][0] - self.arm_info['r'][1] - self.arm_info['r'][2]
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc

        xy21_ = a2xy_ + np.array([-np.cos(a3tr), np.sin(a3tr)]) * self.bar_thc
        xy22_ = a2xy_ + np.array([np.cos(a3tr), -np.sin(a3tr)]) * self.bar_thc
        xy31 = a3xy_ + np.array([np.cos(a3tr), -np.sin(a3tr)]) * self.bar_thc
        xy32 = a3xy_ + np.array([-np.cos(a3tr), np.sin(a3tr)]) * self.bar_thc

        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
        self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))
        self.arm3.vertices = np.concatenate((xy21_, xy22_, xy31, xy32))


class Round(pyglet.window.Window):
    bar_thc = 3

    def __init__(self, arm_info, goal):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Round, self).__init__(width=400, height=400, resizable=False, caption='flat', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.arm_info = arm_info
        self.goal_info = goal
        # self.obstacle_info = obstacle
        self.center_coord = np.array([200, 200])
        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,                # location
                     goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
            ('c3B', (86, 109, 249) * 4))    # color
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,                # location
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))    # color

    def render(self):
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_arm(self):
        # update goal
        self.goal.vertices = (
            200 + self.goal_info['x'] - self.goal_info['l']/2, 200 + self.goal_info['y'] - self.goal_info['l']/2,
            200 + self.goal_info['x'] + self.goal_info['l']/2, 200 + self.goal_info['y'] - self.goal_info['l']/2,
            200 + self.goal_info['x'] + self.goal_info['l']/2, 200 + self.goal_info['y'] + self.goal_info['l']/2,
            200 + self.goal_info['x'] - self.goal_info['l']/2, 200 + self.goal_info['y'] + self.goal_info['l']/2)

        # update arm
        a1l = np.fabs(self.arm_info['l'][0])     # radius, arm length
        # print(a1l)
        a1r = self.arm_info['r'][0]    # radian, angle
        a1xy = self.center_coord            # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy   # a1 end and a2 start (x1, y1)

        a1tr= np.pi / 2 - self.arm_info['r'][0]
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))



if __name__ == '__main__':
    env = ArmEnv()
    for _ in  range(200):
        env.reset()
        for _ in  range(200):
            env.render()
            env.step(env.sample_action())