import os
import time
import visdom
import numpy as np
from utils import tensor2im
from PIL import Image


class Visualizer():

    def __init__(self, opt):

        self.display_id = opt.display_id
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.ncols = opt.display_ncols

        if self.display_id > 0:
            self.vis = visdom.Visdom()

        self.save_img_path = os.path.join(opt.checkpoint_dir, opt.name, 'images')
        if not os.path.exists(self.save_img_path):
            os.makedirs(self.save_img_path)

        self.loss_log_txt = os.path.join(opt.checkpoint_dir, opt.name, 'log_loss.txt')
        with open(self.loss_log_txt, 'a') as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def display_current_results(self, visual_results, epoch):

        if self.display_id > 0:
            ncols = self.ncols
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visual_results))
                h, w = next(iter(visual_results.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name  # monet2photo_cyclegan
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visual_results.items():  # image: 1 x 3 x 256 x 256
                    image_numpy = tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))  # c x h x w
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win=self.display_id + 2,
                              opts=dict(title=title + ' labels'))
        for label, image in visual_results.items():
            image_numpy = tensor2im(image)
            img_path = os.path.join(self.save_img_path, 'epoch%.3d_%s.png' % (epoch, label))
            image_pil = Image.fromarray(image_numpy)
            h, w, _ = image_numpy.shape
            image_pil.save(img_path)

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.loss_log_txt, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message