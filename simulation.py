import numpy as np

# aca esta la simulacion, donde el robot se va moviendo de a poco
# la idea es ir restando alpha*gradiente en cada paso (eso es basicamente
# descenso de gradiente, como cuando se busca el minimo de una funcion)


def simular_trayectoria(x0, y0, xg, yg, fdx, fdy, alpha, max_iter, tol):
    # esta funcion mueve al robot paso a paso y guarda por donde fue pasando
    # para poder graficar la trayectoria despues en streamlit

    trayectoria = [(x0, y0)]
    gradientes = []

    x_act = x0
    y_act = y0

    mensaje = ""
    estado = ""

    llego = False

    for i in range(int(max_iter)):

        gx = float(fdx(x_act, y_act))
        gy = float(fdy(x_act, y_act))
        gradientes.append((gx, gy))

        # si el robot se acerca mucho a un obstaculo el gradiente se va a infinito
        # (porque dividimos por d^2 y d se hace casi 0), entonces hay que chequear
        # esto si no el programa explota con nan en todos lados
        if not np.isfinite(gx) or not np.isfinite(gy):
            mensaje = "El robot se acerco demasiado a un obstaculo (iteracion " + str(i) + "), el potencial se va a infinito ahi y no se puede seguir."
            estado = "colision"
            break

        dist_meta = np.sqrt((x_act - xg)**2 + (y_act - yg)**2)

        if dist_meta < tol:
            mensaje = "Llego a la meta en " + str(i) + " iteraciones."
            estado = "exito"
            llego = True
            break

        norma_grad = np.sqrt(gx**2 + gy**2)

        # si el gradiente es casi cero pero todavia estamos lejos de la meta
        # significa que quedamos atrapados en un minimo local, esto pasa cuando
        # un obstaculo esta justo en el camino hacia la meta (lo vimos varias veces
        # probando con distintas posiciones)
        if norma_grad < 1e-3 and dist_meta > tol * 3:
            mensaje = "El robot quedo atrapado en un minimo local (iteracion " + str(i) + "), el gradiente es casi 0 pero no llegamos a la meta."
            estado = "minimo_local"
            break

        # actualizacion de posicion, formula del descenso de gradiente
        x_sig = x_act - alpha * gx
        y_sig = y_act - alpha * gy

        trayectoria.append((x_sig, y_sig))
        x_act = x_sig
        y_act = y_sig

    if not llego and estado == "":
        # si no entro a ninguno de los if de arriba es porque se acabaron
        # las iteraciones sin pasar nada de lo anterior
        mensaje = "Se llegaron a las " + str(max_iter) + " iteraciones maximas sin llegar a la meta."
        estado = "max_iter"

    return np.array(trayectoria), np.array(gradientes), estado, mensaje
