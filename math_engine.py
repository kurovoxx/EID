import sympy as sp
import numpy as np

def create_symbolic_potentials(xg, yg, obstacles, ka, kr, d_min=0.01):
    """
    Crea las expresiones simbólicas de los potenciales atractivos y repulsivos.
    
    Args:
        xg, yg: Coordenadas de la meta.
        obstacles: Lista de tuplas (xo, yo, radio_influencia).
        ka: Constante de atracción.
        kr: Constante de repulsión.
        d_min: Distancia mínima para evitar divisiones por cero.
    """
    x, y = sp.symbols('x y')
    
    # Potencial Atractivo: U_att = (ka/2) * [(x - xg)^2 + (y - yg)^2]
    u_att = (ka / 2) * ((x - xg)**2 + (y - yg)**2)
    
    # Potencial Repulsivo: sum_i U_rep_i
    u_rep_total = 0
    for xo, yo, radio in obstacles:
        # Distancia al cuadrado: d^2 = (x - xo)^2 + (y - yo)^2
        dist_sq = (x - xo)**2 + (y - yo)**2
        
        # U_rep = kr / dist_sq
        u_rep_i = kr / dist_sq
        u_rep_total += u_rep_i
        
    u_total = u_att + u_rep_total
    
    return x, y, u_att, u_rep_total, u_total

def calculate_gradients(u_total, x, y):
    """Calcula el gradiente simbólico de U_total."""
    grad_x = sp.diff(u_total, x)
    grad_y = sp.diff(u_total, y)
    return grad_x, grad_y

def calculate_hessian(u_total, x, y):
    """Calcula la matriz Hessiana simbólica."""
    fxx = sp.diff(u_total, x, 2)
    fyy = sp.diff(u_total, y, 2)
    fxy = sp.diff(u_total, x, y)
    return fxx, fyy, fxy

def get_numerical_functions(u_total, grad_x, grad_y, x, y):
    """Convierte expresiones simbólicas en funciones numéricas evaluables con NumPy."""
    # Usamos 'numpy' explícitamente y nos aseguramos de que no haya divisiones por cero
    f_u = sp.lambdify((x, y), u_total, modules=['numpy'])
    f_grad_x = sp.lambdify((x, y), grad_x, modules=['numpy'])
    f_grad_y = sp.lambdify((x, y), grad_y, modules=['numpy'])
    return f_u, f_grad_x, f_grad_y

def directional_derivative(u_total, x, y, x0, y0, ux, uy):
    """
    Calcula la derivada direccional en (x0, y0) en la dirección (ux, uy).
    u debe ser un vector unitario.
    """
    grad_x, grad_y = calculate_gradients(u_total, x, y)
    
    # Evaluar gradiente en el punto
    gx_val = grad_x.subs({x: x0, y: y0})
    gy_val = grad_y.subs({x: x0, y: y0})
    
    # Du U = grad U . u
    return float(gx_val * ux + gy_val * uy)

def linearization(u_total, x, y, x0, y0):
    """
    Calcula la aproximación lineal (plano tangente) en (x0, y0).
    L(x,y) = U(x0,y0) + Ux(x0,y0)*(x-x0) + Uy(x0,y0)*(y-y0)
    """
    u_val = float(u_total.subs({x: x0, y: y0}))
    grad_x, grad_y = calculate_gradients(u_total, x, y)
    gx_val = float(grad_x.subs({x: x0, y: y0}))
    gy_val = float(grad_y.subs({x: x0, y: y0}))
    
    L = u_val + gx_val * (x - x0) + gy_val * (y - y0)
    return L
