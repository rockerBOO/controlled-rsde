import torch
from . import unconditional_vector_field, euler_integrate_forward


def dpm_solver_integrate_reverse(
    y_1, original_img, model, eta=0.5, steps=20, prompt="", order=2
):
    """
    Integrate the controlled reverse ODE using DPMSolver++.

    Args:
        y_1: Structured noise tensor [B, C, H, W]
        original_img: Reference image tensor [B, C, H, W]
        model: Pre-trained Flux model
        eta: Controller guidance parameter
        steps: Number of function evaluations
        prompt: Text prompt for editing
        order: Order of DPMSolver++ (1, 2, or 3)

    Returns:
        The edited image
    """
    device = y_1.device
    x = y_1.clone()

    # Initialize time steps (using log-space for better performance)
    # DPMSolver++ works better with log-SNR scheduling
    timesteps = torch.linspace(0, 1, steps + 1, device=device)

    # Storage for model outputs at different orders
    model_output_list = []
    time_list = []

    # DPMSolver++ algorithm
    for i in range(steps):
        # Current and next time
        t = timesteps[i]
        t_next = timesteps[i + 1]

        # Get unconditional reverse vector field
        v_t = -unconditional_vector_field(x, 1 - t, model, prompt)

        # Conditional vector field for reverse process
        v_t_cond = (original_img - x) / (1.0 - t)

        # Controlled reverse vector field from Equation (15)
        controlled_drift = v_t + eta * (v_t_cond - v_t)

        # Store for multi-step methods
        model_output_list.append(controlled_drift)
        time_list.append(t)

        # Calculate step based on DPMSolver++ method
        if len(model_output_list) >= order:
            x = dpm_solver_update(x, model_output_list, time_list, t_next, order)
            # Keep only what's needed for next step
            if order == 1:
                model_output_list = []
                time_list = []
            else:
                model_output_list = model_output_list[-(order - 1) :]
                time_list = time_list[-(order - 1) :]
        else:
            # First-order Euler when we don't have enough previous steps
            dt = t_next - t
            x = x + controlled_drift * dt

    return x


def dpm_solver_update(x, model_output_list, time_list, t_next, order):
    """
    Update step for DPMSolver++.

    Args:
        x: Current state
        model_output_list: List of previous model outputs
        time_list: List of previous time steps
        t_next: Next time step
        order: Order of the solver (1, 2, or 3)

    Returns:
        Updated state
    """
    if order == 1:
        # First-order update (equivalent to Euler)
        return x + (t_next - time_list[-1]) * model_output_list[-1]

    elif order == 2:
        # Second-order update
        t, t_prev = time_list[-1], time_list[-2]
        dt = t_next - t
        dt_prev = t - t_prev

        # Compute coefficients
        r = dt / dt_prev

        # Second-order update formula
        D1 = model_output_list[-1]
        D2 = model_output_list[-2]

        # Second-order DPM-Solver++ update
        x_next = x + dt * ((1 + r) * D1 - r * D2)
        return x_next

    elif order == 3 and len(model_output_list) >= 3:
        # Third-order update
        t, t_prev, t_prev_2 = time_list[-1], time_list[-2], time_list[-3]
        dt = t_next - t
        dt_prev = t - t_prev
        dt_prev_2 = t_prev - t_prev_2

        # Compute coefficients for third-order method
        r1 = dt / dt_prev
        r2 = dt / (dt_prev + dt_prev_2)

        # Get model outputs
        D1 = model_output_list[-1]
        D2 = model_output_list[-2]
        D3 = model_output_list[-3]

        # Third-order DPM-Solver++ update
        coeff_1 = (1 + r1 * (1 + r2)) / (r1 * r2)
        coeff_2 = -r2 * (1 + r1) / (r1 * (r2 - r1))
        coeff_3 = r1 / (r2 * (r2 - r1))

        x_next = x + dt * (coeff_1 * D1 + coeff_2 * D2 + coeff_3 * D3)
        return x_next

    else:
        # Fallback to first-order if we don't have enough steps
        return x + (t_next - time_list[-1]) * model_output_list[-1]


def dpm_solver_integrate_with_schedule(
    y_1, original_img, flux_model, eta_schedule, steps=20, prompt="", order=2
):
    """
    Integrate using DPMSolver++ with a time-varying eta schedule.

    Args:
        y_1: Structured noise tensor [B, C, H, W]
        original_img: Reference image tensor [B, C, H, W]
        flux_model: Pre-trained Flux model
        eta_schedule: Dict with start_step, stop_step, and value
        steps: Number of function evaluations
        prompt: Text prompt for editing
        order: Order of DPMSolver++ (1, 2, or 3)

    Returns:
        The edited image
    """
    device = y_1.device
    x = y_1.clone()

    # Parse schedule parameters
    start_step = eta_schedule.get("start_step", 0)
    stop_step = eta_schedule.get("stop_step", steps)
    eta_value = eta_schedule.get("value", 0.5)

    # Initialize time steps
    timesteps = torch.linspace(0, 1, steps + 1, device=device)

    # Storage for model outputs at different orders
    model_output_list = []
    time_list = []

    # DPMSolver++ algorithm
    for i in range(steps):
        # Current and next time
        t = timesteps[i]
        t_next = timesteps[i + 1]

        # Determine eta value based on current step
        if i >= start_step and i < stop_step:
            eta = eta_value
        else:
            eta = 0.0

        # Get unconditional reverse vector field
        v_t = -unconditional_vector_field(x, 1 - t, flux_model, prompt)

        # Conditional vector field for reverse process
        v_t_cond = (original_img - x) / (1.0 - t)

        # Controlled reverse vector field
        controlled_drift = v_t + eta * (v_t_cond - v_t)

        # Store for multi-step methods
        model_output_list.append(controlled_drift)
        time_list.append(t)

        # Calculate step based on DPMSolver++ method
        if len(model_output_list) >= order:
            x = dpm_solver_update(x, model_output_list, time_list, t_next, order)
            # Keep only what's needed for next step
            if order == 1:
                model_output_list = []
                time_list = []
            else:
                model_output_list = model_output_list[-(order - 1) :]
                time_list = time_list[-(order - 1) :]
        else:
            # First-order when we don't have enough previous steps
            dt = t_next - t
            x = x + controlled_drift * dt

    return x


def invert_and_edit_with_dpm_solver(
    original_img,
    model,
    edit_prompt,
    gamma=0.5,
    eta=0.5,
    inversion_steps=50,
    editing_steps=20,
    order=2,
    eta_schedule=None,
):
    """
    Full pipeline for inverting and editing using DPMSolver++.

    Args:
        original_img: Input image tensor [B, C, H, W]
        model: Pre-trained Flux model
        edit_prompt: Text prompt for desired edit
        gamma: Controller guidance for inversion
        eta: Controller guidance for editing
        inversion_steps: Steps for inversion
        editing_steps: Steps for editing (can be fewer with DPMSolver++)
        order: Order of DPMSolver++
        eta_schedule: Optional schedule for eta

    Returns:
        Edited image
    """
    # 1. Inversion - still use Euler for accuracy
    structured_noise = euler_integrate_forward(
        original_img, model, gamma, inversion_steps
    )

    # 2. Editing - use DPMSolver++ for efficiency
    if eta_schedule is None:
        edited_img = dpm_solver_integrate_reverse(
            structured_noise,
            original_img,
            model,
            eta,
            editing_steps,
            edit_prompt,
            order,
        )
    else:
        edited_img = dpm_solver_integrate_with_schedule(
            structured_noise,
            original_img,
            model,
            eta_schedule,
            editing_steps,
            edit_prompt,
            order,
        )

    return edited_img
