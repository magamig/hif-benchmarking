import numpy as np
import os
import ntpath
import time
from . import util
from . import html

# from scipy.misc import imresize
from skimage.transform import resize
import skimage.measure as ski_measure
import collections
import pickle
import scipy.io as io

# save image to the disk


def get_random_point(img, scale_factor):
    img_c, img_h, img_w = img.shape
    """two random point position in low resolution image """
    low_point1_h = np.random.randint(0, img_h)
    low_point1_w = np.random.randint(0, img_w)
    # low_point2_h = np.random.randint(0,img_h)
    # low_point2_w = np.random.randint(0,img_w)
    """corresponding position in low resolution image"""
    high_point1_h = low_point1_h * scale_factor
    high_point1_w = low_point1_w * scale_factor
    # high_point2_h = low_point2_h*scale_factor
    # high_point2_w = low_point2_w*scale_factor
    # return {'1':[low_point1_h, low_point1_w],
    #         '2':[low_point2_h, low_point2_w]},{'1':[high_point1_h, high_point1_w],
    #                                            '2':[high_point2_h, high_point2_w]}
    return {"1": [low_point1_h, low_point1_w]}, {"1": [high_point1_h, high_point1_w]}


def convert2samesize(image_list):
    img_c = image_list[0].shape[0]
    height_max = np.array([img.shape[1] for img in image_list]).max()
    weight_max = np.array([img.shape[2] for img in image_list]).max()
    return [resize(img, (img_c, height_max, weight_max)) for img in image_list]


def get_spectral_lines(real_img, rec_img, points):
    lines = {}
    for key, value in points.items():
        lines[key] = [real_img[:, value[0], value[1]], rec_img[:, value[0], value[1]]]
    return lines


def paint_point_in_img(img, points):
    assert len(img.shape) == 3
    for key, value in points.items():
        img[:, value[0] - 5 : value[0] + 5, value[1] - 5 : value[1] + 5] = 1
    return img


def compute_sam(x_true, x_pred):
    assert x_true.ndim == 3 and x_true.shape == x_pred.shape

    w, h, c = x_true.shape
    x_true = x_true.reshape(c, -1)
    x_pred = x_pred.reshape(c, -1)

    sam = (x_true * x_pred).sum(axis=1) / (
        np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1)
    )

    sam = np.arccos(sam) * 180 / np.pi
    mSAM = sam.mean()
    var_sam = np.var(sam)
    return mSAM, var_sam


def compute_psnr(img1, img2):
    assert img1.ndim == 3 and img2.ndim == 3

    # n_bands = img1.shape[0]
    # psnr_list = [ski_measure.compare_psnr(img1[i,:,:], img2[i,:,:]) for i in range(n_bands)]
    # var_psnr = np.var(psnr_list)
    # mpsnr = np.mean(np.array(psnr_list))
    # return mpsnr, var_psnr
    img_c, img_w, img_h = img1.shape
    ref = img1.reshape(img_c, -1)
    tar = img2.reshape(img_c, -1)
    msr = np.mean((ref - tar) ** 2, 1)
    max2 = np.max(ref, 1) ** 2
    # import ipdb
    # ipdb.set_trace()
    psnrall = 10 * np.log10(max2 / msr)
    out_mean = np.mean(psnrall)
    return out_mean


class Visualizer:
    def __init__(self, opt, wavelengh):
        self.wavelengh = wavelengh
        self.display_id = opt.display_id
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt
        self.saved = False
        self.uni_id = 66
        if self.display_id > 0:
            import visdom

            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(
                server=opt.display_server,
                port=opt.display_port,
                env=opt.display_env,
                raise_exceptions=True,
            )

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, "loss_log.txt")
        self.precision_path = os.path.join(
            opt.checkpoints_dir, opt.name, "precision.txt"
        )
        self.save_psnr_sam_path = os.path.join(
            opt.checkpoints_dir, opt.name, "psnr_and_sam.pickle"
        )
        self.save_hhsi_path = os.path.join(opt.checkpoints_dir, opt.name)
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write(
                "================ Training Loss (%s) ================\n" % now
            )
        with open(self.precision_path, "a") as precision_file:
            now = time.strftime("%c")
            precision_file.write(
                "================ Precision Log (%s) ================\n" % now
            )

    def reset(self):
        self.saved = False

    def throw_visdom_connection_error(self):
        print(
            "\n\nCould not connect to Visdom server (https://github.com/facebookresearch/visdom) for displaying training progress.\nYou can suppress connection to Visdom using the option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start the server by \n$ python -m visdom.server.\n\n"
        )
        exit(1)

    # |visuals|: dictionary of images to display or save
    def display_current_results(
        self, visuals, image_name, epoch, save_result, win_id=[1]
    ):
        if self.display_id > 0:  # show images in the browser
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (
                    w,
                    h,
                )
                title = self.name
                label_html = ""
                label_html_row = ""
                images = []

                idx = 0
                for label, image in visuals.items():

                    image_numpy = util.tensor2im(image, self.wavelengh)
                    label_html_row += "<td>%s</td>" % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += "<tr>%s</tr>" % label_html_row
                        label_html_row = ""
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += "<td></td>"
                    idx += 1
                if label_html_row != "":
                    label_html += "<tr>%s</tr>" % label_html_row
                # pane col = image row

                img = images.pop()
                # img = paint_point_in_img(img, points)
                images.append(img)

                try:
                    self.vis.images(
                        convert2samesize(images),
                        nrow=ncols,
                        win=self.display_id + win_id[0],
                        padding=2,
                        opts=dict(title=title + " images"),
                    )
                    label_html = "<table>%s</table>" % label_html

                except ConnectionError:
                    self.throw_visdom_connection_error()

            else:
                idx = 10
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image, self.wavelengh)
                    self.vis.image(
                        image_numpy.transpose([2, 0, 1]),
                        opts=dict(title=label),
                        win=self.display_id + idx,
                    )
                    idx += 1

    def plot_spectral_lines(
        self, visuals, image_name, visual_corresponding_name=None, win_id=None
    ):
        """get image"""
        real_hsi = visuals["real_HrHsi"].data.cpu().float().numpy()[0]
        rec_hsi = (
            visuals[visual_corresponding_name["real_HrHsi"]]
            .data.cpu()
            .float()
            .numpy()[0]
        )
        real_lhsi = visuals["real_LrHsi"].data.cpu().float().numpy()[0]
        rec_lhsi = (
            visuals[visual_corresponding_name["real_LrHsi"]]
            .data.cpu()
            .float()
            .numpy()[0]
        )
        scale_factor = real_hsi.shape[1] // real_lhsi.shape[1]
        """get random two points position for plot spectral lines"""
        low_points, high_points = get_random_point(real_lhsi, scale_factor)
        """high resolution image spectral lines"""
        lines = get_spectral_lines(real_hsi, rec_hsi, high_points)
        len_spectral = np.arange(len(lines["1"][0]))
        self.vis.line(
            Y=np.column_stack(
                [np.column_stack((line[0], line[1])) for line in lines.values()]
            ),
            X=np.column_stack([len_spectral] * 2 * len(lines)),
            win=self.display_id + win_id[0],
            opts=dict(title="spectral"),
        )

        """low resolution image spectral lines"""
        lines = get_spectral_lines(real_lhsi, rec_lhsi, low_points)
        len_spectral = np.arange(len(lines["1"][0]))
        y_column_stack = np.column_stack(
            [np.column_stack((line[0], line[1])) for line in lines.values()]
        )
        self.vis.line(
            Y=y_column_stack,
            X=np.column_stack([len_spectral] * (2 * len(lines))),
            win=self.display_id + win_id[1],
            opts=dict(title="spectral_low_img"),
        )

    def plot_psnr_sam(self, visuals, image_name, epoch, counter_ratio, visual_corresponding_name=None):
        """psnr and sam updating with epoch"""
        real_hsi = visuals["real_HrHsi"].data.cpu().float().numpy()[0]
        rec_hsi = (
            visuals[visual_corresponding_name["real_HrHsi"]]
            .data.cpu()
            .float()
            .numpy()[0]
        )

        if not hasattr(self, "plot_precision"):
            self.plot_precision = {"X": {}, "Y": {}}
            self.win_id_dict = {}

        if image_name[0] not in self.plot_precision["X"]:
            self.plot_precision["X"][image_name[0]] = []
            self.plot_precision["Y"][image_name[0]] = []

        self.plot_precision["X"][image_name[0]].append(
            [epoch + counter_ratio, epoch + counter_ratio]
        )
        result_sam, _ = compute_sam(real_hsi, rec_hsi)
        result_psnr = compute_psnr(real_hsi, rec_hsi)
        self.plot_precision["Y"][image_name[0]].append([result_sam, result_psnr])
        """save txt"""
        write_message = "Epoch:{} Name:{} PSNR:{} SAM:{}".format(
            epoch + counter_ratio, image_name[0], result_psnr, result_sam
        )
        with open(self.precision_path, "a") as precision_file:
            precision_file.write("%s\n" % write_message)

        """plot line"""
        if image_name[0] not in self.win_id_dict:
            self.win_id_dict[image_name[0]] = self.uni_id
            self.uni_id += 1
            print("uni_id", self.uni_id)
        try:
            self.vis.line(
                X=np.column_stack(
                    [np.row_stack(self.plot_precision["X"][image_name[0]])]
                ),
                Y=np.column_stack(
                    [np.row_stack(self.plot_precision["Y"][image_name[0]])]
                ),
                win=self.display_id + self.win_id_dict[image_name[0]],
                opts=dict(title="SAM and psnr of " + image_name[0]),
            )
        except ConnectionError:
            self.throw_visdom_connection_error()

        """save"""
        if not hasattr(self, "sava_precision"):
            self.sava_precision = collections.OrderedDict()
        if image_name[0] not in self.sava_precision:
            self.sava_precision[image_name[0]] = []
        self.sava_precision[image_name[0]].append([result_sam, result_psnr])
        savefiles = open(self.save_psnr_sam_path, "wb")
        pickle.dump(self.sava_precision, savefiles)
        savefiles.close()
        # np.save(os.path.join(self.save_hhsi_path, "real_{}.npy".format(image_name[0])), real_hsi)
        # np.save(os.path.join(self.save_hhsi_path, "rec_{}.npy".format(image_name[0])), rec_hsi)

        io.savemat(
            os.path.join(self.save_hhsi_path, "real_{}.mat".format(image_name[0])),
            {"RealHSI": real_hsi.transpose(1, 2, 0)},
        )
        io.savemat(
            os.path.join(self.save_hhsi_path, "rec_{}.mat".format(image_name[0])),
            {"RecHSI": rec_hsi.transpose(1, 2, 0)},
        )

    # losses: dictionary of error labels and values
    def plot_current_losses(self, epoch, counter_ratio, opt, losses):
        if not hasattr(self, "plot_data"):
            self.plot_data = {"X": [], "Y": [], "legend": list(losses.keys())}
        self.plot_data["X"].append(epoch + counter_ratio)
        self.plot_data["Y"].append([losses[k] for k in self.plot_data["legend"]])
        try:
            self.vis.line(
                X=np.stack(
                    [np.array(self.plot_data["X"])] * len(self.plot_data["legend"]), 1
                ),
                Y=np.array(self.plot_data["Y"]),
                opts={
                    "title": self.name + " loss over time",
                    "legend": self.plot_data["legend"],
                    "xlabel": "epoch",
                    "ylabel": "loss",
                },
                win=self.display_id,
            )
        except ConnectionError:
            self.throw_visdom_connection_error()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, losses, t):
        message = "(epoch: %d, iters: %d, time: %.3f) " % (epoch, i, t)
        for k, v in losses.items():
            message += "%s: %.3f " % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write("%s\n" % message)

    def plot_lr(self, lr, epoch):
        if not hasattr(self, "lr"):
            self.lr = {"X": [], "Y": []}

        self.lr["X"].append(epoch)
        self.lr["Y"].append(lr)
        try:
            self.vis.line(
                X=np.array(self.lr["X"]),
                Y=np.array(self.lr["Y"]),
                opts={"title": "learning rate", "xlabel": "epoch", "ylabel": "lr"},
                win=78,
            )
        except ConnectionError:
            self.throw_visdom_connection_error()
