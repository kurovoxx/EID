import numpy as np

def run_simulation(x0, y0, xg, yg, f_grad_x, f_grad_y, alpha, max_iter, tol, obstacles):
    """
    Simula la trayectoria del robot usando descenso de gradiente.
    
    Returns:
        history: Lista de posiciones (x, y).
        status: Diccionario con tipo de estado, mensaje descriptivo y métricas finales.
    """
    history = [(x0, y0)]
    grads = [] # Lista de (gx, gy)
    norms = [] # Lista de ||grad||
    
    curr_x, curr_y = x0, y0
    
    # Inicialización del resultado con valores por defecto
    result = {
        "type": "running", 
        "message": "Simulando...", 
        "final_grad": (0.0, 0.0), 
        "grad_norm": 0.0
    }
    
    for i in range(max_iter):
        try:
            # Intentar calcular gradiente
            gx_raw = f_grad_x(curr_x, curr_y)
            gy_raw = f_grad_y(curr_x, curr_y)
            
            gx = float(gx_raw)
            gy = float(gy_raw)
            grad_norm = np.sqrt(gx**2 + gy**2)
            
            grads.append((gx, gy))
            norms.append(grad_norm)
            
            # Actualizar métricas finales en tiempo real
            result["final_grad"] = (gx, gy)
            result["grad_norm"] = grad_norm
            
            dist_to_goal = np.sqrt((curr_x - xg)**2 + (curr_y - yg)**2)

            # 1. Verificar estabilidad numérica (NaN/Inf)
            if not np.isfinite(gx) or not np.isfinite(gy):
                result.update({
                    "type": "error",
                    "message": "⚠️ Error numérico: El gradiente ha explotado (posible colisión directa)."
                })
                break

            # 2. Condición de éxito
            if dist_to_goal < tol:
                result.update({
                    "type": "success",
                    "message": f"✅ ¡Objetivo alcanzado en {i} iteraciones!"
                })
                break

            # 3. Detección de Mínimo Local
            if grad_norm < 1e-3 and dist_to_goal > tol * 2:
                result.update({
                    "type": "local_minimum",
                    "message": f"🛑 Atrapado en un mínimo local. El gradiente es casi nulo."
                })
                break
            
            # 4. Actualización de posición
            next_x = curr_x - alpha * gx
            next_y = curr_y - alpha * gy
            
            # 5. Detección de estancamiento
            step_size = np.sqrt((next_x - curr_x)**2 + (next_y - curr_y)**2)
            if step_size < 1e-7 and dist_to_goal > tol:
                 result.update({
                    "type": "stuck",
                    "message": "⚠️ El robot se ha detenido por falta de fuerza en el campo."
                })
                 break

            history.append((next_x, next_y))
            curr_x, curr_y = next_x, next_y
            
        except Exception as e:
            result.update({
                "type": "error",
                "message": f"❌ Error durante el cálculo: {str(e)}"
            })
            break
            
    else:
        result.update({
            "type": "max_iter",
            "message": f"⏳ Se alcanzó el máximo de {max_iter} iteraciones."
        })
        
    return np.array(history), result, np.array(grads), np.array(norms)
