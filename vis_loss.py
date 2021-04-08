# coding:utf8
# import visdom
# import time
# import numpy as np
#
#
# class Visualizer(object):
#     def __init__(self, env='default', **kwargs):
#         self.vis = visdom.Visdom(env=env, **kwargs)
#         self.index = {}
#     def plot_many_stack(self, d):
#         '''
#         self.plot('loss',1.00)
#         '''
#         name = list(d.keys())
#         name_total = " ".join(name)
#         x = self.index.get(name_total, 0)
#         val = list(d.values())
#         if len(val) == 1:
#             y = np.array(val)
#         else:
#             y = np.array(val).reshape(-1, len(val))
#         # print(x)
#         self.vis.line(Y=y, X=np.ones(y.shape) * x,
#                       win=str(name_total),  # unicode
#                       opts=dict(legend=name,
#                                 title=name_total),
#                       update=None if x == 0 else 'append'
#                       )
#         self.index[name_total] = x + 1


from visdom import Visdom
import numpy as np
import time

class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis1 = Visdom()
        self.vis2 = Visdom()
        self.vis3 = Visdom()
        self.vis4 = Visdom()
        self.vis5 = Visdom()
        self.vis6 = Visdom()
        self.vis1.line([0.], [0], win='d_loss', opts=dict(title='d_loss'))
        self.vis2.line([0.], [0], win='g_loss', opts=dict(title='g_loss'))
        self.vis3.line([0.], [0], win='adv_loss', opts=dict(title='adv_loss'))
        self.vis4.line([0.], [0], win='cycle_loss', opts=dict(title='cycle_loss'))
        self.vis5.line([0.], [0], win='identity_loss', opts=dict(title='identity_loss'))
        self.vis6.line([0.], [0], win='cam_loss', opts=dict(title='cam_loss'))

    def plot_many_stack(self, global_steps, d_loss, g_loss, adv_loss, cycle_loss, identity_loss, cam_loss):

        d_loss = np.log(d_loss.item())
        g_loss = np.log(g_loss.item())
        adv_loss = np.log(adv_loss.item())
        cycle_loss = np.log(cycle_loss.item())
        identity_loss = np.log(identity_loss.item())
        cam_loss = np.log(cam_loss.item())

        self.vis1.line([d_loss], [global_steps], win='d_loss', update='append')
        self.vis2.line([g_loss], [global_steps], win='g_loss', update='append')
        self.vis3.line([adv_loss], [global_steps], win='adv_loss', update='append')
        self.vis4.line([cycle_loss], [global_steps], win='cycle_loss', update='append')
        self.vis5.line([identity_loss], [global_steps], win='identity_loss', update='append')
        self.vis6.line([cam_loss], [global_steps], win='cam_loss', update='append')
        # time.sleep(0.5)
