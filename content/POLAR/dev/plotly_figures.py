from __future__ import annotations
from typing import Union, Sequence
import logging
from functools import wraps
import torch
from torch import Tensor
import numpy as np
from scipy.io import loadmat
import plotly.express as px
import plotly.graph_objects as go


def safe_unbatch(function):
    """ A simple decorator that allows function that are designed to take batches of data and
        parameters as input to accept single data and param as well.
        It basically unsqueezes data and params, and squeeze back the result(s).
    """
    is_sequence = lambda x: hasattr(x, '__iter__') and not isinstance(x, str)

    @wraps(function)
    def batch_safe_function(data, *args, **kwargs):
        nonlocal function
        if data.ndim == 3:
            return function(data, *args, **kwargs)
        # ugly special cases
        if function.__name__ in ['add_outliers', 'translate', 'dig']:
            args = tuple([(x, ) for x in args])
            return function(data.unsqueeze(0), *args, **kwargs).squeeze(0)
        # Tuple do not support item assignment so args are casted to list, processed, and casted
        # back to tuple. Maybe there's a better way to do ?
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, Tensor):
                args[i] = arg.unsqueeze(0)
                continue
            if is_sequence(arg):
                args[i] = (arg, )
        args = tuple(args)
        for k, v in kwargs.items():
            if isinstance(v, Tensor):
                kwargs[k] = v.unsqueeze(0)
                continue
            if is_sequence(v):
                kwargs[k] = (v, )
        batch = data.unsqueeze(0)
        result = function(batch, *args, **kwargs)
        if isinstance(result, Tensor):
            return result.squeeze(0)
        result = list(result)
        for i, res in enumerate(result):
            result[i] = res.squeeze()
        result = tuple(result)
        return result

    return batch_safe_function


def multibackend(function):
    """ For a function that acts on Tensor, this decorator allows to take numpy arrays as well.
    If so, they are casted to Tensor, and casted back to numpy arrays after the function execution.
    """
    @wraps(function)
    def wrapper(*args, **kwargs):
        to_numpy = False
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                args[i] = torch.as_tensor(arg)
                to_numpy = True
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                kwargs[k] = torch.as_tensor(v)
                to_numpy = True
        output = function(*args, **kwargs)
        if not isinstance(output, tuple):
            return output.detach().cpu().numpy() if to_numpy else output
        output = list(output)
        for i, out in enumerate(output):
            if isinstance(out, torch.Tensor) and to_numpy is True:
                output[i] = out.detach().cpu().numpy()
        return output
    return wrapper


@multibackend
@safe_unbatch
def center(pointclouds: Tensor) -> Tensor:
    """ Center a batch of point clouds on the origin.

    Args:
        pointclouds (Tensor): Batch of point clouds of shape `(batch_size, num_points, *)` where
                              `*` denotes spatial coordinates.

    Returns:
        center: Batch of point clouds of shape `(batch_size, num_points, *)` where `*` denotes
                spatial coordinates, centered on the origin. That is, the mean over the second axis
                is `(0, 0, 0)`.
    """
    return pointclouds - pointclouds.mean(axis=1, keepdim=True)


@multibackend
@safe_unbatch
def scale(pointclouds: Tensor, factors: Union[float, Sequence[float]] = 1.0) -> Tensor:
    """ Multiply a batch of points coordinates by a common factor if `factors` is a float or
    by a specific value for each point cloud if `factors` is a sequence.

    Args:
        pointclouds (Tensor): Batch of point clouds of shape `(batch_size, num_points, *)` where
                              `*` denotes spatial coordinates.
        factors (Union[float, Sequence[float]], optional):
            Factor by which to multiply each point clouds.

    Returns:
        scaled_batch: Batch of point clouds of shape `(batch_size, num_points, *)` where `*`
                      denotes spatial coordinates, where each point cloud has been scaled by the
                      given factor.
    """
    factors = torch.as_tensor(factors, device=pointclouds.device)
    if factors.ndim == 0:
        factors = factors.repeat(len(pointclouds))
    return pointclouds * factors.unsqueeze(dim=1)[:, :, None]


@multibackend
@safe_unbatch
def normalize(pointclouds: Tensor) -> Tensor:
    """ Centralize a batch of point clouds and divide each of them by its maximal norm.

    Args:
        pointclouds (Tensor): Batch of point clouds of shape `(batch_size, num_points, *)` where
                              `*` denotes spatial coordinates.

    Returns:
        normalized_batch: Normalized batch of point clouds of shape `(batch_size, num_points, *)`
                          where `*` denotes spatial coordinates
    """
    pointclouds = center(pointclouds)
    max_norm = pointclouds.norm(dim=2).amax(dim=1)
    return scale(pointclouds, 1 / max_norm)


def get_all_named_cmaps() -> list[str]:
    named_colorscales = px.colors.named_colorscales()
    base_cmaps = ["cividis", "sunset", "turbo", "thermal"]
    base_cmaps_idx = [named_colorscales.index(x) for x in base_cmaps]
    for idx in base_cmaps_idx:
        named_colorscales.pop(idx)
    named_colorscales = base_cmaps + named_colorscales
    return named_colorscales


def interactive_plot(
    data: Union[np.ndarray, Sequence[np.ndarray]],
    labels: Union[str, Sequence[str]] = None,
    point_size: int = 3,
    colorbar: bool = False,
    color: np.ndarray = None,
    color_range: tuple[int] = None,
    cmap: str = None,
    constraint_x: bool = False,
    constraint_y: bool = False,
    constraint_z: bool = False,
    return_fig: bool = False,
    width: int = 500,
    height: int = 500,
    title: str = None,
    buttons: bool = True,
) -> Union[None, go.Figure]:
    """ Interactive plot of point cloud(s) based on Plotly. Can display N pointcloud(s).

    Args:
        pointcloud (Union[np.ndarray, tuple[np.ndarray]]): If a list or tuple is passed,
        each element will be displayed with its own name, colormap, and button to toggle visibility.
        label (str, optional): This will be the title of the plot. Is is called label because
            it was intended to be used within a classification setup. Defaults to None.
        point_size (int, optional): Display size of one point (x, y, z). Defaults to 1.
        color (np.ndarray, optional): Color of each points. MUST be a sequence of length equals
            to the number of points. If None, the z coordinates will be used to color points.
            Defaults to None.
        constraint_x (bool, optional): Rescale x within [-1 ,1]. Defaults to False.
        constraint_y (bool, optional): Rescale y within [-1 ,1]. Defaults to False.
        constraint_z (bool, optional): Rescale z within [-1 ,1]. Defaults to False.

    Raises:
        ValueError: If too few or too many pointclouds are passed to the function.
    """
    if not isinstance(data, (list, tuple)):
        data = [data]
    if isinstance(data, tuple):
        data = list(data)
    # move to cpu if required; raises warning
    moved_to_cpu = False
    for i, x in enumerate(data):
        if isinstance(x, Tensor) and x.is_cuda:
            data[i] = x.cpu()
            moved_to_cpu = True
    if moved_to_cpu:
        logging.basicConfig(format='[%(levelname)s] %(message)s')
        logging.warning((
            "CUDA Tensors were passed in: I had to move them to the cpu in order to display them."
        ))
    N = len(data)
    if labels is None:
        labels = [f"pointcloud {i + 1}" for i in range(N)]
    if not isinstance(labels, (list, tuple)):
        labels = (labels, )
    if labels is not None and not len(data) == len(labels):
        raise ValueError(f"You gave {len(data)} pointclouds but {len(labels)} labels.")
    all_cmaps = get_all_named_cmaps()
    if isinstance(cmap, str):
        cmaps = N * [cmap]
    elif not (isinstance(cmap, (tuple, list)) and len(cmap) == len(data)):
        cmaps = all_cmaps[:N]
    else:
        cmaps = cmap
    traces = list()
    for pointcloud, label, cmap in zip(data, labels, cmaps):
        x, y, z = pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2]
        c = color if color is not None else z
        marker_kwargs = dict(size=point_size, opacity=0.8, color=c, colorscale=cmap)
        if color_range is not None:
            marker_kwargs['cmin'] = color_range[0]
            marker_kwargs['cmax'] = color_range[1]
        if colorbar:
            marker_kwargs["colorbar"] = dict(thickness=20)
        scatter_kwargs = dict(visible=True, mode='markers', name=label, marker=marker_kwargs)
        traces.append(go.Scatter3d(x=x, y=y, z=z, **scatter_kwargs))
    layout = dict(
        width=width, height=height,
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]), margin=dict(t=50)
    )
    if constraint_x:
        layout['scene'] = dict(xaxis=dict(nticks=4, range=[-1, 1]))
    if constraint_y:
        layout['scene'] = dict(yaxis=dict(nticks=4, range=[-1, 1]))
    if constraint_z:
        layout['scene'] = dict(zaxis=dict(nticks=4, range=[-1, 1]))
    fig = go.Figure(data=traces)
    fig.update_layout(**layout)
    if N > 1 and buttons:
        allButton = dict(
            method='restyle',
            label='all',
            visible=True,
            args=[{'visible': True}],
            args2=[{'visible': 'legendonly'}]
        )
        buttons = list()
        for i in range(N):
            label = labels[i]
            traces_idx = [i for i, x in enumerate(traces) if x.name == label]
            button = dict(
                method='restyle',
                label=label,
                visible=True,
                args=[{'visible': True}, traces_idx],
                args2=[{'visible': 'legendonly'}, traces_idx]
            )
            buttons.append(button)
        # create the layout
        layout = go.Layout(
            updatemenus=[
                dict(
                    type='buttons',
                    direction='right',
                    x=1.4,
                    y=1.1,
                    showactive=True,
                    buttons=[allButton, *buttons]
                )
            ],
            showlegend=True
        )
        fig.update_layout(layout)
    if title is not None:
        fig.update_layout(title=title)
    if return_fig:
        return fig
    fig.show()
