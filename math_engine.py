import sympy as sp
import numpy as np

# en este archivo esta toda la parte matematica, las funciones de potencial
# y las derivadas. usamos sympy pa que las derivadas salgan bien y no haya
# que sacarlas a mano (intentamos a mano primero pero nos tinco mal en un obstaculo
# asi que mejor que lo haga el sympy)

x, y = sp.symbols('x y')


def potencial_atractivo(xg, yg, ka):
    # esta es la formula que dan en el pdf del proyecto
    # U_att = (ka/2)*((x-xg)^2+(y-yg)^2)
    # basicamente es como un hoyo, el minimo de esto esta justo en la meta (xg,yg)
    return (ka / 2) * ((x - xg)**2 + (y - yg)**2)


def potencial_repulsivo(obstaculos, kr):
    # cada obstaculo suma kr/d^2 (d = distancia del robot al obstaculo)
    # entre mas cerca este el robot del obstaculo esto se dispara para arriba
    # ojo: si el robot pasa justo encima de un obstaculo esto explota (d=0), eso
    # despues nos genero problemas en la simulacion (revisar mas abajo en simulation.py)
    total = 0
    for obs in obstaculos:
        xo = obs[0]
        yo = obs[1]
        d2 = (x - xo)**2 + (y - yo)**2
        total = total + kr / d2
    return total


def potencial_total(xg, yg, obstaculos, ka, kr):
    # junta todo, el potencial que realmente usa el robot para moverse
    Uatt = potencial_atractivo(xg, yg, ka)
    Urep = potencial_repulsivo(obstaculos, kr)
    U = Uatt + Urep
    return Uatt, Urep, U


def calcular_gradiente(U):
    # derivadas parciales, esto es lo que pide el enunciado en la parte 6.3 y 6.4
    dUdx = sp.diff(U, x)
    dUdy = sp.diff(U, y)
    return dUdx, dUdy


def pasar_a_funcion_numerica(U, dUdx, dUdy):
    # ojo aca: sympy es lento para evaluar en muchos puntos (probamos para hacer
    # la grilla del grafico 3d directo con sympy y se quedaba pegado como 1 minuto)
    # con lambdify se pasa a una funcion de python normal y evalua rapido con numpy
    fU = sp.lambdify((x, y), U, 'numpy')
    fdx = sp.lambdify((x, y), dUdx, 'numpy')
    fdy = sp.lambdify((x, y), dUdy, 'numpy')
    return fU, fdx, fdy


def derivada_direccional(dUdx, dUdy, px, py, vx, vy):
    # derivada direccional = gradiente . vector unitario
    # primero hay que normalizar el vector (que quede de largo 1)
    norma = np.sqrt(vx**2 + vy**2)
    if norma == 0:
        return None  # no se puede dividir por cero, el vector (0,0) no tiene direccion
    ux = vx / norma
    uy = vy / norma

    gx = float(dUdx.subs({x: px, y: py}))
    gy = float(dUdy.subs({x: px, y: py}))

    resultado = gx * ux + gy * uy
    return resultado


def plano_tangente(U, dUdx, dUdy, px, py):
    # esto es la aproximacion lineal (lo que en clase llamamos linealizacion)
    # L(x,y) = U(p) + Ux(p)*(x-px) + Uy(p)*(y-py)
    Up = float(U.subs({x: px, y: py}))
    gx = float(dUdx.subs({x: px, y: py}))
    gy = float(dUdy.subs({x: px, y: py}))

    L = Up + gx * (x - px) + gy * (y - py)
    return L, Up, gx, gy


def buscar_punto_critico_cerca(dUdx, dUdy, x0, y0):
    # esta funcion busca un punto donde el gradiente = (0,0), o sea un punto critico
    # usamos el metodo de newton que vimos en clase de calculo numerico (clase del profe)
    # IMPORTANTE: esto no siempre encuentra el punto, si el punto inicial que se le
    # da esta muy lejos puede no converger. probamos varios puntos iniciales antes de
    # que funcionara bien, ojito con eso al usarlo

    # la hessiana (matriz de segundas derivadas)
    fxx = sp.diff(dUdx, x)
    fxy = sp.diff(dUdx, y)
    fyy = sp.diff(dUdy, y)

    xi = x0
    yi = y0

    encontro = False
    for i in range(50):
        gx_val = float(dUdx.subs({x: xi, y: yi}))
        gy_val = float(dUdy.subs({x: xi, y: yi}))

        if abs(gx_val) < 1e-6 and abs(gy_val) < 1e-6:
            encontro = True
            break

        a = float(fxx.subs({x: xi, y: yi}))
        b = float(fxy.subs({x: xi, y: yi}))
        d = float(fyy.subs({x: xi, y: yi}))

        det = a * d - b * b
        if abs(det) < 1e-9:
            # esto significa que la hessiana no se puede invertir, no podemos seguir
            return None

        # paso de newton (sacamos la formula de la matriz inversa 2x2, esta en el cuaderno)
        dx_paso = -(d * gx_val - b * gy_val) / det
        dy_paso = -(-b * gx_val + a * gy_val) / det

        xi = xi + dx_paso
        yi = yi + dy_paso

    if not encontro:
        # se acabaron las 50 vueltas y no convergio, mejor decir que no se encontro
        # nada en vez de devolver cualquier cosa que este mal
        return None

    # ahora clasificamos el punto con el criterio de la segunda derivada (visto en clases)
    a = float(fxx.subs({x: xi, y: yi}))
    b = float(fxy.subs({x: xi, y: yi}))
    d = float(fyy.subs({x: xi, y: yi}))
    det = a * d - b * b

    if det > 0 and a > 0:
        tipo = "Minimo local"
    elif det > 0 and a < 0:
        tipo = "Maximo local"
    elif det < 0:
        tipo = "Punto de silla"
    else:
        tipo = "No se pudo clasificar (caso borde)"

    return {"x": xi, "y": yi, "tipo": tipo, "det_hessiano": det}
