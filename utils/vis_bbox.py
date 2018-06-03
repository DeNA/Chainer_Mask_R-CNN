from chainercv.visualizations.vis_image import vis_image
import numpy as np
from skimage.measure import find_contours
from matplotlib.patches import Polygon
import cv2

def vis_bbox(img, bbox, label=None, score=None, mask=None, label_names=None, ax=None, contour=False, labeldisplay=True):
    """Visualize bounding boxes inside image.

    Example:

        >>> from chainercv.datasets import VOCDetectionDataset
        >>> from chainercv.datasets import voc_bbox_label_names
        >>> from chainercv.visualizations import vis_bbox
        >>> import matplotlib.pyplot as plot
        >>> dataset = VOCDetectionDataset()
        >>> img, bbox, label = dataset[60]
        >>> vis_bbox(img, bbox, label,
        ...         label_names=voc_bbox_label_names)
        >>> plot.show()

    Args:
        img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`.
        bbox (~numpy.ndarray): An array of shape :math:`(R, 4)`, where
            :math:`R` is the number of bounding boxes in the image.
            Each element is organized
            by :obj:`(y_min, x_min, y_max, x_max)` in the second axis.
        label (~numpy.ndarray): An integer array of shape :math:`(R,)`.
            The values correspond to id for label names stored in
            :obj:`label_names`. This is optional.
        score (~numpy.ndarray): A float array of shape :math:`(R,)`.
             Each value indicates how confident the prediction is.
             This is optional.
        label_names (iterable of strings): Name of labels ordered according
            to label ids. If this is :obj:`None`, labels will be skipped.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """
    from matplotlib import pyplot as plot

    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')

    # alpha-blend the masks
    COLOR=[(1,1,0), (1,0,1),(0,1,1),(0,0,1),(0,1,0), (1,0,0),(0.1,1,0.2)]
    dst = img.astype(float)
    for i, m in enumerate(mask):
        alpha = np.tile(np.round(m), (3, 1, 1)).astype(float) * 0.4
        src1 = np.ones(dst.shape).astype(float)
        for j, col in enumerate(COLOR[i%len(COLOR)]):
            src1[j] *= col * 255
        dst = cv2.multiply(src1, alpha) + cv2.multiply(dst, 1 - alpha)

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(dst, ax=ax)

    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax

    # add boxes, contours and labels
    for i, bb in enumerate(bbox):
        # boxes
        xy = (bb[1], bb[0])
        height = int(bb[2]) - int(bb[0])
        width = int(bb[3]) - int(bb[1])
        ax.add_patch(plot.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=1))
        
        # contours
        if contour:
            Mcontours = find_contours(mask[i].T, 0.5)
            for verts in Mcontours:
                p = Polygon(verts, facecolor="none", edgecolor=[0.5,0.5,0.5])
                ax.add_patch(p)
        
        #labels
        caption = list()
        if label is not None and label_names is not None:
            lb = label[i]
            print(lb)
            if not (0 <= lb < len(label_names)):
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0 and labeldisplay:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    fontsize=8,
                    color='white'
                    )#'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
    return ax
