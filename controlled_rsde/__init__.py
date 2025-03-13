import torch
from torchdiffeq import odeint


def unconditional_vector_field(y_t, t, model, prompt=""):
    """
    Computes the unconditional vector field ut(Yt) using the Flux model.

    Args:
        y_t: Current state tensor [B, C, H, W]
        t: Current time step (scalar between 0 and 1)
        model: Pre-trained Flux model
        prompt: Text prompt (empty for inversion)

    Returns:
        Vector field at current state and time
    """
    # Get the prompt embedding
    prompt_embedding = model.get_text_embeddings(prompt)

    # The Flux model provides the vector field
    return model(y_t, t, prompt_embedding)


def conditional_vector_field(y_t, t, y_1):
    """
    Computes the conditional vector field ut(Yt|y1) based on Proposition 3.2.

    Args:
        y_t: Current state tensor [B, C, H, W]
        t: Current time step (scalar between 0 and 1)
        y_1: Target noise tensor [B, C, H, W]

    Returns:
        Conditional vector field at current state and time
    """
    # Formula from Proposition 3.2: (y1 - yt)/(1-t)
    return (y_1 - y_t) / (1.0 - t)


def controlled_vector_field(y_t, t, model, y_1, gamma, prompt=""):
    """
    Computes the controlled vector field for ODE (8).

    Args:
        y_t: Current state tensor [B, C, H, W]
        t: Current time step (scalar between 0 and 1)
        model: Pre-trained Flux model
        y_1: Target noise tensor [B, C, H, W]
        gamma: Controller guidance parameter
        prompt: Text prompt (empty for inversion)

    Returns:
        Controlled vector field at current state and time
    """
    # Compute the unconditional vector field
    u_t = unconditional_vector_field(y_t, t, model, prompt)

    # Compute the conditional vector field
    u_t_cond = conditional_vector_field(y_t, t, y_1)

    # Combine them according to Equation (8)
    return u_t + gamma * (u_t_cond - u_t)




# Example of using the continuous schedule
def linear_decay_eta(step, total_steps):
    """Linearly decay eta from 1.0 to 0.0 over the first 30% of steps"""
    decay_point = int(0.3 * total_steps)

    if step < decay_point:
        return 1.0 - (step / decay_point)
    else:
        return 0.0


def controlled_ode_function(t, y, flux_model, y_1, gamma, prompt=""):
    """ODE function for torchdiffeq solver"""
    return controlled_vector_field(y, t, flux_model, y_1, gamma, prompt)


def integrate_with_odeint(y_0, flux_model, gamma=0.5, steps=100, noise_sample=None):
    """Use torchdiffeq for more accurate integration"""
    # Sample a target noise if not provided
    if noise_sample is None:
        y_1 = torch.randn_like(y_0)
    else:
        y_1 = noise_sample

    # Create time points
    t = torch.linspace(0, 1, steps, device=y_0.device)

    # Define ODE function with fixed parameters
    def ode_func(t, y):
        return controlled_ode_function(t, y, flux_model, y_1, gamma)

    # Solve ODE
    solution = odeint(ode_func, y_0, t, method="rk4")

    # Return final state
    return solution[-1]
