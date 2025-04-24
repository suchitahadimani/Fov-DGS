#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.rigid_utils import from_homogenous, to_homogenous
from torchvision.transforms.functional import gaussian_blur


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, d_xyz, d_rotation, d_scaling, is_6dof=False,
           scaling_modifier=1.0, override_color=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_densify = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_densify.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if is_6dof:
        if torch.is_tensor(d_xyz) is False:
            means3D = pc.get_xyz
        else:
            means3D = from_homogenous(
                torch.bmm(d_xyz, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1))
    else:
        means3D = pc.get_xyz + d_xyz
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling + d_scaling
        rotations = pc.get_rotation + d_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        means2D_densify=screenspace_points_densify,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "viewspace_points_densify": screenspace_points_densify,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth}


import torch
import math
from gaussian_renderer import render
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.rigid_utils import from_homogenous, to_homogenous
from torchvision.transforms.functional import gaussian_blur
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as R


def compute_eccentricity(gaze_point, gaussian_center, image_size):
    P = gaze_point / image_size.to(device=gaze_point.device)
    mu = gaussian_center / image_size.to(device=gaussian_center.device)
    return torch.norm(P - mu, dim=-1)

def compute_acuity(w0, m, e):
    return w0 + m * e

def compute_layer(L, acuity, w0, m, e0):
    if not torch.is_tensor(acuity):
        acuity = torch.tensor(acuity, dtype=torch.float32)
    denominator = w0 + m * e0
    denominator = torch.tensor(denominator, dtype=torch.float32)
    lod = torch.ceil(torch.tensor(L, dtype=torch.float32) - torch.log2(acuity / denominator))
    return lod.clamp(0, L - 1).long()

def compute_f_from_csf(acuity, sal, k):
    return acuity / (1 + k * sal)

def create_gaussian_filter(cov_matrix, f, image_size, scale=1.0):
    H, W = image_size
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, steps=H),
        torch.linspace(-1, 1, steps=W),
        indexing="ij"
    )
    grid = torch.stack([x, y], dim=-1)

    mu = torch.tensor([0.0, 0.0])
    diff = grid - mu

    identity = torch.eye(2)
    cov_f = cov_matrix + (1 / f) * identity
    cov_inv = torch.inverse(cov_f)

    diff_flat = diff.view(-1, 2)
    mdist = torch.einsum("bi,ij,bj->b", diff_flat, cov_inv, diff_flat)
    gaussian = torch.exp(-0.5 * mdist).view(H, W)

    gaussian /= gaussian.sum()
    return gaussian

def generate_saliency_map(coarse_render):
    gray = torch.mean(coarse_render, dim=0, keepdim=True)  # (1, H, W)
    saliency = torch.abs(torch.nn.functional.conv2d(
        gray.unsqueeze(0),
        weight=torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], device=gray.device),
        padding=1
    )).squeeze(0).squeeze(0)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    return saliency


def render_foveated(viewpoint_camera, pc, forest, pipe, bg_color, d_xyz, d_rotation, d_scaling, gaze_point,
                    is_6dof=False, scaling_modifier=1.0, override_color=None, L=2, w0=1/48, m=1.32, e0=0.1, k=0.4,
                    saliency_map=None):
    print("RENDER FOVEATED", flush=True)
    device = pc.get_xyz.device
    H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
    image_size = torch.tensor([W, H], device=device)

    rendered = None
    all_radii = torch.zeros((pc.get_xyz.shape[0]), device=device)
    all_visibility = torch.zeros((pc.get_xyz.shape[0]), dtype=torch.bool, device=device)
    screenspace_points_all = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, device=device)
    screenspace_points_densify_all = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, device=device)
    depth_map = torch.zeros((H, W), device=device)

    indices_layers = []

    dynamic_forest = [tree for tree in forest if any(n.Dyn for n in tree)]
    static_forest = [tree for tree in forest if all(not n.Dyn for n in tree)]

    def get_projected_mu(indices_tensor):
        projected_2d = pc.get_projected_2d(
            viewpoint_camera.world_view_transform,
            viewpoint_camera.full_proj_transform,
            W, H
        )[indices_tensor]
        return projected_2d.mean(dim=0)  # 2D center of the tree

    # ---------- DYNAMIC FOREST (Two-phase deformation) ----------
    for tree in dynamic_forest:
        if not tree:
            continue
        indices = [n.index for n in tree if 0 <= n.index < pc.get_xyz.shape[0]]
        if not indices:
            continue

        indices_tensor = torch.tensor(indices, device=device)
        mu = get_projected_mu(indices_tensor)
        e = compute_eccentricity(gaze_point, mu.unsqueeze(0), image_size).item()
        acuity = compute_acuity(w0, m, e)
        layer_1 = compute_layer(L, acuity, w0, m, e0).item()
        layer_1 = torch.clamp(torch.tensor(layer_1), 0, L - 1).long().item()

        # Apply deformation to Gaussians in layer_1
        l1_indices = [n.index for n in tree if n.layer == layer_1]
        if not l1_indices:
            continue
        l1_tensor = torch.tensor(l1_indices, device=device)
        dx_1 = d_xyz[l1_tensor] if d_xyz is not None else 0
        dr_1 = d_rotation[l1_tensor] if d_rotation is not None else 0
        ds_1 = d_scaling[l1_tensor] if d_scaling is not None else 0

        # Recompute layer selection (Phase 2)
        mu = get_projected_mu(l1_tensor)
        e = compute_eccentricity(gaze_point, mu.unsqueeze(0), image_size).item()
        sal = saliency_map[int(mu[1]), int(mu[0])] if saliency_map is not None else 0.0
        acuity = compute_acuity(w0, m, e)
        f = compute_f_from_csf(acuity, sal, k)
        layer_2 = compute_layer(L, acuity, w0, m, e0).item()
        layer_2 = torch.clamp(torch.tensor(layer_2), 0, L - 1).long().item()

        layers_to_render = [layer_2] if layer_2 == L - 1 else [layer_2, layer_2 + 1]

        for l in layers_to_render:
            final_indices = [n.index for n in tree if n.layer == l]
            if not final_indices:
                continue
            final_tensor = torch.tensor(final_indices, device=device)
            if final_tensor.max().item() >= d_xyz.shape[0] or final_tensor.min().item() < 0:
                print(f"[ERROR] Out-of-bounds index: min={final_tensor.min().item()}, max={final_tensor.max().item()}, allowed=0 to {d_xyz.shape[0] - 1}")
                continue

            dx = d_xyz[final_tensor] if d_xyz is not None else 0
            dr = d_rotation[final_tensor] if d_rotation is not None else 0
            ds = d_scaling[final_tensor] if d_scaling is not None else 0
            indices_layers.append((final_tensor, dx, dr, ds))

    # ---------- STATIC FOREST (Only CSF selection) ----------
    for tree in static_forest:
        if not tree:
            continue
        indices = [n.index for n in tree if 0 <= n.index < pc.get_xyz.shape[0]]
        if not indices:
            continue

        indices_tensor = torch.tensor(indices, device=device)
        mu = get_projected_mu(indices_tensor)
        e = compute_eccentricity(gaze_point, mu.unsqueeze(0), image_size).item()
        sal = saliency_map[int(mu[1]), int(mu[0])] if saliency_map is not None else 0.0
        acuity = compute_acuity(w0, m, e)
        f = compute_f_from_csf(acuity, sal, k)
        layer = compute_layer(L, acuity, w0, m, e0).item()
        layer = torch.clamp(torch.tensor(layer), 0, L - 1).long().item()

        layers_to_render = [layer] if layer == L - 1 else [layer, layer + 1]

        for l in layers_to_render:
            final_indices = [n.index for n in tree if n.layer == l]
            if not final_indices:
                continue
            final_tensor = torch.tensor(final_indices, device=device)
            # No deformation for static
            dx = dr = ds = 0
            indices_layers.append((final_tensor, dx, dr, ds))

    # ---------- Final Batched Render ----------
    if indices_layers:
        all_indices = torch.cat([x[0] for x in indices_layers], dim=0)
        valid_mask = (all_indices >= 0) & (all_indices < pc.get_xyz.shape[0])
        if not valid_mask.all():
            print("[WARNING] Some indices were out-of-bounds, skipping them.")
            all_indices = all_indices[valid_mask]


        if all_indices.numel() > 0:
            pc_l = pc.get_filtered_copy(all_indices)
            dx_all = torch.cat([x[1] if isinstance(x[1], torch.Tensor) else torch.zeros((x[0].shape[0], 3), device=device) for x in indices_layers], dim=0)
            #dr_all = torch.cat([x[2] if isinstance(x[2], torch.Tensor) else torch.zeros((x[0].shape[0], 3), device=device) for x in indices_layers], dim=0)
            ds_all = torch.cat([x[3] if isinstance(x[3], torch.Tensor) else torch.zeros((x[0].shape[0], 3), device=device) for x in indices_layers], dim=0)
            dr_axis_angle = torch.cat([
                x[2] if isinstance(x[2], torch.Tensor) else torch.zeros((x[0].shape[0], 3), device=device)
                for x in indices_layers
            ], dim=0)  # [N, 3]

            # Convert to numpy
            dr_np = dr_axis_angle.detach().cpu().numpy()

            # Axis-angle → quaternion (scipy returns [x, y, z, w])
            quats_xyzw = R.from_rotvec(dr_np).as_quat()  # [N, 4]

            # Reorder to [w, x, y, z]
            quats_wxyz = np.concatenate([quats_xyzw[:, 3:4], quats_xyzw[:, 0:3]], axis=1)

            # Back to torch
            dr_all = torch.tensor(quats_wxyz, dtype=torch.float32, device=device)


            if is_6dof:
                N = dx_all.shape[0]
                d_xyz_full = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1)
                d_xyz_full[:, :3, 3] = dx_all
                d_xyz_full[:, 0, 0] *= (1.0 + ds_all[:, 0])
                d_xyz_full[:, 1, 1] *= (1.0 + ds_all[:, 1])
                d_xyz_full[:, 2, 2] *= (1.0 + ds_all[:, 2])
            else:
                d_xyz_full = dx_all
            


            try:
                results = render(viewpoint_camera, pc_l, pipe, bg_color, d_xyz_full, dr_all, ds_all, is_6dof, scaling_modifier)
                rendered = results["render"]
                all_radii[all_indices] = results["radii"]
                all_visibility[all_indices] = results["visibility_filter"]
                screenspace_points_all[all_indices] = results["viewspace_points"]
                screenspace_points_densify_all[all_indices] = results["viewspace_points_densify"]
                depth_map = torch.maximum(depth_map, results["depth"])
            except torch.cuda.OutOfMemoryError:
                print(f"[OOM] Combined render OOM for {all_indices.shape[0]} points", flush=True)
                rendered = None

    print("RENDERING DONE", flush=True)

    if rendered is None:
        print("[WARNING] No Gaussians rendered, using default black image.")
        rendered = torch.zeros((3, H, W), device=device)
        screenspace_points_all = torch.zeros((pc.get_xyz.shape[0], 3), device=device)
        screenspace_points_densify_all = torch.zeros((pc.get_xyz.shape[0], 3), device=device)
        all_visibility = torch.zeros((pc.get_xyz.shape[0]), dtype=torch.bool, device=device)
        all_radii = torch.zeros((pc.get_xyz.shape[0]), device=device)
        depth_map = torch.zeros((H, W), device=device)

    return {
        "render": rendered,
        "viewspace_points": screenspace_points_all,
        "viewspace_points_densify": screenspace_points_densify_all,
        "visibility_filter": all_visibility,
        "radii": all_radii,
        "depth": depth_map
    }

def render_foveated_static_only(viewpoint_camera, pc, forest, pipe, bg_color, gaze_point,
                                scaling_modifier=1.0, L=2, w0=1/48, m=1.32, e0=0.1, k=0.4,
                                saliency_map=None):
    print("RENDER FOVEATED (STATIC ONLY)", flush=True)
    device = pc.get_xyz.device
    H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
    image_size = torch.tensor([W, H], device=device)

    rendered = None
    all_radii = torch.zeros((pc.get_xyz.shape[0]), device=device)
    all_visibility = torch.zeros((pc.get_xyz.shape[0]), dtype=torch.bool, device=device)
    screenspace_points_all = torch.zeros_like(pc.get_xyz, device=device)
    screenspace_points_densify_all = torch.zeros_like(pc.get_xyz, device=device)
    depth_map = torch.zeros((H, W), device=device)

    indices_layers = []

    static_forest = [tree for tree in forest if all(not n.Dyn for n in tree)]

    def get_projected_mu(indices_tensor):
        projected_2d = pc.get_projected_2d(
            viewpoint_camera.world_view_transform,
            viewpoint_camera.full_proj_transform,
            W, H
        )[indices_tensor]
        return projected_2d.mean(dim=0)  # 2D center of the tree

    for tree in static_forest:
        if not tree:
            continue

        indices = [n.index for n in tree if 0 <= n.index < pc.get_xyz.shape[0]]
        if not indices:
            continue

        indices_tensor = torch.tensor(indices, device=device)
        mu = get_projected_mu(indices_tensor)
        e = compute_eccentricity(gaze_point, mu.unsqueeze(0), image_size).item()
        sal = saliency_map[int(mu[1]), int(mu[0])] if saliency_map is not None else 0.0
        acuity = compute_acuity(w0, m, e)
        f = compute_f_from_csf(acuity, sal, k)
        layer = compute_layer(L, acuity, w0, m, e0).item()
        layer = torch.clamp(torch.tensor(layer), 0, L - 1).long().item()

        layers_to_render = [layer] if layer == L - 1 else [layer, layer + 1]

        for l in layers_to_render:
            final_indices = [n.index for n in tree if n.layer == l]
            if not final_indices:
                continue
            final_tensor = torch.tensor(final_indices, device=device)
            indices_layers.append(final_tensor)

    # Combine and render
    if indices_layers:
        all_indices = torch.cat(indices_layers, dim=0)
        valid_mask = (all_indices >= 0) & (all_indices < pc.get_xyz.shape[0])
        all_indices = all_indices[valid_mask]

        if all_indices.numel() > 0:
            pc_l = pc.get_filtered_copy(all_indices)
            zeros = torch.zeros_like(pc_l.get_xyz)

            try:
                results = render(viewpoint_camera, pc_l, pipe, bg_color, zeros, zeros, zeros,
                                 is_6dof=False, scaling_modifier=scaling_modifier)

                rendered = results["render"]
                all_radii[all_indices] = results["radii"]
                all_visibility[all_indices] = results["visibility_filter"]
                screenspace_points_all[all_indices] = results["viewspace_points"]
                screenspace_points_densify_all[all_indices] = results["viewspace_points_densify"]
                depth_map = torch.maximum(depth_map, results["depth"])
            except torch.cuda.OutOfMemoryError:
                print(f"[OOM] Failed to render {all_indices.shape[0]} points", flush=True)
                rendered = None

    if rendered is None:
        print("[WARNING] No Gaussians rendered, using black image.")
        rendered = torch.zeros((3, H, W), device=device)

    return {
        "render": rendered,
        "viewspace_points": screenspace_points_all,
        "viewspace_points_densify": screenspace_points_densify_all,
        "visibility_filter": all_visibility,
        "radii": all_radii,
        "depth": depth_map
    }

'''
def render_foveated(viewpoint_camera, pc, forest, pipe, bg_color, d_xyz, d_rotation, d_scaling, gaze_point,
                    is_6dof=False, scaling_modifier=1.0, override_color=None, L=2, w0=1/48, m=1.32, e0=0.1, k=0.4,
                    saliency_map=None):
    print("RENDER FOVEATED", flush=True)
    device = pc.get_xyz.device
    H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
    image_size = torch.tensor([W, H], device=device)
    MAX_BATCH_SIZE = 10000

    rendered = None
    N = pc.get_xyz.shape[0]
    all_radii = torch.zeros((N,), device=device)
    all_visibility = torch.zeros((N,), dtype=torch.bool, device=device)
    screenspace_points_all = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, device=device)
    screenspace_points_densify_all = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, device=device)
    depth_map = torch.zeros((H, W), device=device)

    batch_indices = []
    batch_dynamics = []

    def render_batch(indices, dynamics):
        nonlocal rendered, all_radii, all_visibility, screenspace_points_all, screenspace_points_densify_all, depth_map

        indices_tensor = torch.tensor(indices, device=device)
        pc_l = pc.get_filtered_copy(indices_tensor)
        if pc_l.get_xyz.shape[0] == 0:
            return

        is_dynamic = any(dynamics)
        dx = d_xyz[indices_tensor] if is_dynamic and d_xyz is not None else 0
        dr = d_rotation[indices_tensor] if is_dynamic and d_rotation is not None else 0
        ds = d_scaling[indices_tensor] if is_dynamic and d_scaling is not None else 0

        try:
            results = render(viewpoint_camera, pc_l, pipe, bg_color, dx, dr, ds, is_6dof, scaling_modifier)

            if rendered is None:
                rendered = results["render"]
            else:
                rendered = rendered + results["render"].detach()

            all_radii[indices_tensor] = results["radii"].to(all_radii.dtype)
            all_visibility[indices_tensor] = results["visibility_filter"].to(all_visibility.dtype)
            screenspace_points_all[indices_tensor] = results["viewspace_points"].to(screenspace_points_all.dtype)
            screenspace_points_densify_all[indices_tensor] = results["viewspace_points_densify"].to(
                screenspace_points_densify_all.dtype)
            depth_map = torch.maximum(depth_map, results["depth"])

            del results, pc_l, dx, dr, ds
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"[OOM] Skipping chunk of {len(indices)} points", flush=True)
            torch.cuda.empty_cache()

    # === Batch all nodes from the forest ===
    for tree in forest:
        for node in tree:
            if 0 <= node.index < N:
                batch_indices.append(node.index)
                batch_dynamics.append(node.Dyn)

            if len(batch_indices) >= MAX_BATCH_SIZE:
                render_batch(batch_indices, batch_dynamics)
                batch_indices.clear()
                batch_dynamics.clear()

    # Render any remaining nodes
    if batch_indices:
        render_batch(batch_indices, batch_dynamics)

    print("RENDERING DONE", flush=True)

    if rendered is None:
        print("[WARNING] No Gaussians rendered, using default black image.")
        rendered = torch.zeros((3, H, W), device=device)

    return {
        "render": rendered,
        "viewspace_points": screenspace_points_all,
        "viewspace_points_densify": screenspace_points_densify_all,
        "visibility_filter": all_visibility,
        "radii": all_radii,
        "depth": depth_map
    }




def render_foveated(viewpoint_camera, pc, forest, pipe, bg_color, d_xyz, d_rotation, d_scaling, gaze_point,
                    is_6dof=False, scaling_modifier=1.0, override_color=None, L=2, w0=1/48, m=1.32, e0=0.1, k=0.4,
                    saliency_map=None):
    print("RENDER FOVEATED", flush=True)
    device = pc.get_xyz.device
    H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
    image_size = torch.tensor([W, H], device=device)
    MAX_BATCH_SIZE = 5000

    rendered = None
    all_radii = torch.zeros((pc.get_xyz.shape[0]), device=device)
    all_visibility = torch.zeros((pc.get_xyz.shape[0]), dtype=torch.bool, device=device)
    screenspace_points_all = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, device=device)
    screenspace_points_densify_all = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, device=device)
    depth_map = torch.zeros((H, W), device=device)

    dynamic_forest = [tree for tree in forest if any(n.Dyn for n in tree)]
    static_forest = [tree for tree in forest if all(not n.Dyn for n in tree)]

    for tree in dynamic_forest + static_forest:
        if not tree:
            continue

        is_dynamic = any(n.Dyn for n in tree)
        indices = [n.index for n in tree if 0 <= n.index < pc.get_xyz.shape[0]]

        if not indices:
            continue

        indices_tensor = torch.tensor(indices, device=device)

        # Compute eccentricity to determine foveated LOD (optional, could remove if not needed)
        projected_2d = pc.get_projected_2d(
            viewpoint_camera.world_view_transform,
            viewpoint_camera.full_proj_transform,
            viewpoint_camera.image_width,
            viewpoint_camera.image_height
        )[indices_tensor]

        mu = projected_2d.mean(dim=0)
        e = compute_eccentricity(gaze_point, mu.unsqueeze(0), image_size).item()
        acuity = compute_acuity(w0, m, e)
        fovea_layer = compute_layer(L, acuity, w0, m, e0).item()
        fovea_layer = torch.clamp(torch.tensor(fovea_layer, dtype=torch.float32), 0, L - 1).long().item()

        # (Optional) skip trees with far LODs
        # if fovea_layer > X: continue

        for chunk in torch.split(indices_tensor, MAX_BATCH_SIZE):
            pc_l = pc.get_filtered_copy(chunk)
            if pc_l.get_xyz.shape[0] == 0:
                continue

            if is_dynamic:
                dx = d_xyz[chunk] if d_xyz is not None else 0
                dr = d_rotation[chunk] if d_rotation is not None else 0
                ds = d_scaling[chunk] if d_scaling is not None else 0
            else:
                dx = dr = ds = 0

            try:
                results = render(viewpoint_camera, pc_l, pipe, bg_color, dx, dr, ds, is_6dof, scaling_modifier)

                if rendered is None:
                    rendered = results["render"]
                else:
                    rendered += results["render"].detach()

                all_radii[chunk] = results["radii"].to(all_radii.dtype)
                all_visibility[chunk] = results["visibility_filter"].to(all_visibility.dtype)
                screenspace_points_all[chunk] = results["viewspace_points"].to(screenspace_points_all.dtype)
                screenspace_points_densify_all[chunk] = results["viewspace_points_densify"].to(
                    screenspace_points_densify_all.dtype)
                depth_map = torch.maximum(depth_map, results["depth"])

                del results, pc_l, dx, dr, ds
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"[OOM] Skipping chunk of {chunk.shape[0]} points from tree with {len(indices)} points", flush=True)
                torch.cuda.empty_cache()
                continue

    print("RENDERING DONE", flush=True)

    if rendered is None:
        print("[WARNING] No Gaussians rendered, using default black image.")
        rendered = torch.zeros((3, H, W), device=device)
        screenspace_points_all = torch.zeros((pc.get_xyz.shape[0], 3), device=device)
        screenspace_points_densify_all = torch.zeros((pc.get_xyz.shape[0], 3), device=device)
        all_visibility = torch.zeros((pc.get_xyz.shape[0]), dtype=torch.bool, device=device)
        all_radii = torch.zeros((pc.get_xyz.shape[0]), device=device)
        depth_map = torch.zeros((H, W), device=device)

    return {
        "render": rendered,
        "viewspace_points": screenspace_points_all,
        "viewspace_points_densify": screenspace_points_densify_all,
        "visibility_filter": all_visibility,
        "radii": all_radii,
        "depth": depth_map
    }



def render_foveated(viewpoint_camera, pc, forest, pipe, bg_color, d_xyz, d_rotation, d_scaling, gaze_point, is_6dof=False, scaling_modifier=1.0, override_color=None, L=2, w0=1/48, m=1.32, e0=0.1, k=0.4, saliency_map=None):
    print("RENDER FOVEATED", flush=True)
    device = pc.get_xyz.device
    H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
    image_size = torch.tensor([W, H], device=device)

    rendered = None
    all_radii = torch.zeros((pc.get_xyz.shape[0]), device=device)
    all_visibility = torch.zeros((pc.get_xyz.shape[0]), dtype=torch.bool, device=device)
    screenspace_points_all = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, device=device)
    screenspace_points_densify_all = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, device=device)
    depth_map = torch.zeros((H, W), device=device)

    dynamic_forest = [tree for tree in forest if any(n.Dyn for n in tree)]
    static_forest = [tree for tree in forest if all(not n.Dyn for n in tree)]

    for tree in dynamic_forest + static_forest:
        if not tree:
            continue
        

        is_dynamic = any(n.Dyn for n in tree)
        nodes = torch.tensor([n.index for n in tree], device=device)
        if torch.any(nodes >= pc.get_xyz.shape[0]) or torch.any(nodes < 0):
            print(f"[WARNING] Skipping tree with invalid node indices: max={nodes.max().item()}, min={nodes.min().item()}, total={len(nodes)}", flush=True)
            continue

        projected_2d = pc.get_projected_2d(
            viewpoint_camera.world_view_transform,
            viewpoint_camera.full_proj_transform,
            viewpoint_camera.image_width,
            viewpoint_camera.image_height
        )[nodes]

        mu = projected_2d.mean(dim=0)
        e = compute_eccentricity(gaze_point, mu.unsqueeze(0), image_size).item()
        acuity = compute_acuity(w0, m, e)
        l1 = compute_layer(L, acuity, w0, m, e0).item()
        l1 = torch.clamp(torch.tensor(l1, dtype=torch.float32), 0, L - 1).long().item()

        layers_to_render = [l1] if l1 == L - 1 else [l1, l1 + 1]

        for l in layers_to_render:
            
            indices = [n.index for n in tree if n.layer == l]
            #print(f"[Layer {l}] Rendering {len(indices)} points in tree {tree} of size {len(tree)}", flush=True)
            if not indices:
                continue

            indices_tensor = torch.tensor(indices, device=device)
            pc_l = pc.get_filtered_copy(indices_tensor)

            if pc_l.get_xyz.shape[0] == 0:
                continue
            if pc_l.get_xyz.shape[0] > 100000:
                continue

            if is_dynamic:
                dx = d_xyz[indices_tensor] if d_xyz is not None else 0
                dr = d_rotation[indices_tensor] if d_rotation is not None else 0
                ds = d_scaling[indices_tensor] if d_scaling is not None else 0
            else:
                dx = dr = ds = 0


            results = render(viewpoint_camera, pc_l, pipe, bg_color, dx, dr, ds, is_6dof, scaling_modifier)

            if rendered is None:
                rendered = results["render"]
            else:
                rendered = rendered + results["render"]

            global_idx = indices_tensor
            all_radii[global_idx] = results["radii"].to(all_radii.dtype)
            all_visibility[global_idx] = results["visibility_filter"].to(all_visibility.dtype)
            screenspace_points_all[global_idx] = results["viewspace_points"].to(screenspace_points_all.dtype)
            screenspace_points_densify_all[global_idx] = results["viewspace_points_densify"].to(screenspace_points_densify_all.dtype)

            depth_map = torch.maximum(depth_map, results["depth"])
            torch.cuda.empty_cache()
            del results


    print("RENDERING DONE", flush=True)
    print(torch.cuda.memory_summary(device="cuda"), flush = True)


    if rendered is None:
        print("[WARNING] No Gaussians rendered, using default black image.")
        rendered = torch.zeros((3, H, W), device=device)

        screenspace_points_all = torch.zeros((pc.get_xyz.shape[0], 3), device=device)
        screenspace_points_densify_all = torch.zeros((pc.get_xyz.shape[0], 3), device=device)
        all_visibility = torch.zeros((pc.get_xyz.shape[0]), dtype=torch.bool, device=device)
        all_radii = torch.zeros((pc.get_xyz.shape[0]), device=device)
        depth_map = torch.zeros((H, W), device=device)

    return {
        "render": rendered,
        "viewspace_points": screenspace_points_all,
        "viewspace_points_densify": screenspace_points_densify_all,
        "visibility_filter": all_visibility,
        "radii": all_radii,
        "depth": depth_map
    }



'''

#if saliency_map is not None and l == l1 + 1:
#    cov2d = pc.get_covariance()[indices_tensor]
#    f_val = compute_f_from_csf(acuity, sal, k)
#    for i, idx in enumerate(indices_tensor):
#        sigma_2d = cov2d[i][:2, :2]
#        kernel = create_gaussian_filter(sigma_2d, f_val, (9, 9))
#        kernel = kernel.to(results["render"].device)
#        kernel = kernel.unsqueeze(0).unsqueeze(0)
#        kernel = kernel.expand(3, 1, *kernel.shape[-2:])
#        render_tensor = results["render"].unsqueeze(0)
#        results["render"] = F.conv2d(render_tensor, kernel, padding=kernel.shape[-1] // 2, groups=3).squeeze(0)
#if saliency_map is not None:
#    mu_pix = mu.long()
#    sal = saliency_map[mu_pix[1], mu_pix[0]].item()
#    f = compute_f_from_csf(acuity, sal, k)