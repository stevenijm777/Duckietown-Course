import matplotlib.pyplot as plt
import numpy as np

def _draw_frame(ax, x, y, theta, label_x, label_y, color='black', scale=0.5):
    """Función auxiliar interna para dibujar ejes de coordenadas."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Vectores dirección
    dx_x, dy_x = scale * cos_t, scale * sin_t
    dx_y, dy_y = scale * np.cos(theta + np.pi/2), scale * np.sin(theta + np.pi/2)
    
    # Eje X
    ax.arrow(x, y, dx_x, dy_x, head_width=0.08, head_length=0.1, fc=color, ec=color)
    ax.text(x + dx_x * 1.2, y + dy_x * 1.2, label_x, color=color, fontsize=12, fontweight='bold')
    
    # Eje Y
    ax.arrow(x, y, dx_y, dy_y, head_width=0.08, head_length=0.1, fc=color, ec=color)
    ax.text(x + dx_y * 1.2, y + dy_y * 1.2, label_y, color=color, fontsize=12, fontweight='bold')

def _draw_duckie_body(ax, x, y, theta, color='orange'):
    """Función auxiliar interna para dibujar el cuerpo del robot."""
    # Triángulo estilizado
    triangle = np.array([[0.2, 0], [-0.1, 0.15], [-0.1, -0.15]])
    
    # Matriz de rotación R
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    # Transformación: Rotación + Traslación
    transformed_triangle = (R @ triangle.T).T + [x, y]
    
    ax.fill(transformed_triangle[:, 0], transformed_triangle[:, 1], color=color, alpha=0.6, label='Cuerpo Robot')
    ax.plot(transformed_triangle[:, 0], transformed_triangle[:, 1], color='black', linewidth=1)
    ax.scatter(x, y, color='black', s=30, zorder=5) # Centro

def visualizar_pose(x, y, theta):
    """
    Función principal para visualizar la pose del robot en el plano 2D.
    Args:
        x (float): Posición en X
        y (float): Posición en Y
        theta (float): Orientación en radianes
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Configuración del plot (limpio)
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')
    
    # Ocultar marcos innecesarios
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    # 1. Dibujar World Frame (Fijo)
    _draw_frame(ax, 0, 0, 0, label_x=r'$X_W$', label_y=r'$Y_W$', color='blue', scale=0.8)
    
    # 2. Dibujar Robot Frame y Cuerpo (Móvil)
    _draw_duckie_body(ax, x, y, theta)
    _draw_frame(ax, x, y, theta, label_x=r'$x^r$', label_y=r'$y^r$', color='darkred', scale=0.4)
    
    # Títulos y leyenda
    degrees = np.degrees(theta)
    plt.title(f"Visualización de Marcos de Referencia\nPose: x={x}, y={y}, $\\theta={degrees:.1f}^\circ$", fontsize=14)
    plt.legend(loc='upper right')
    plt.show()


def visualizar_rotacion_simple(p_world, theta):
    """
    Visualiza un punto y sus proyecciones en los ejes ROTADOS del robot.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    limit = 3.0
    ax.set_aspect('equal')
    ax.set_xlim(-1, limit)
    ax.set_ylim(-1, limit)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 1. Marco del MUNDO (Azul)
    ax.arrow(0, 0, 1.5, 0, head_width=0.1, color='blue', alpha=0.6, width=0.01)
    ax.arrow(0, 0, 0, 1.5, head_width=0.1, color='blue', alpha=0.6, width=0.01)
    ax.text(1.6, 0, "$X_W$", color='blue', fontweight='bold')
    ax.text(0, 1.6, "$Y_W$", color='blue', fontweight='bold')
    
    # 2. Marco del ROBOT (Rojo)
    c, s = np.cos(theta), np.sin(theta)
    # Ejes visuales
    ax.arrow(0, 0, c*1.8, s*1.8, head_width=0.1, color='darkred', width=0.02) # Eje x_r largo
    ax.arrow(0, 0, -s*1.2, c*1.2, head_width=0.1, color='darkred', width=0.01) # Eje y_r
    ax.text(c*1.9, s*1.9, "$x^r$ (Frente)", color='darkred', fontweight='bold', fontsize=10)
    ax.text(-s*1.3, c*1.3, "$y^r$ (Izq)", color='darkred', fontweight='bold', fontsize=10)
    
    # 3. CÁLCULO DE PROYECCIONES (La parte mágica)
    # Queremos saber cuánto avanza en X_robot y luego en Y_robot
    # Proyección escalar sobre el vector unitario del robot
    # Vector unitario X del robot: [c, s]
    distancia_x_robot = p_world[0] * c + p_world[1] * s
    
    # Coordenadas en el MUNDO donde termina la componente X del robot
    # (El punto de la esquina del triángulo rectángulo rotado)
    corner_x = distancia_x_robot * c
    corner_y = distancia_x_robot * s
    
    # 4. DIBUJAR LAS LÍNEAS PUNTEADAS (Proyección en ejes del Robot)
    # Línea Componente X (Desde origen hasta la esquina)
    ax.plot([0, corner_x], [0, corner_y], 
            color='red', linestyle='--', linewidth=2, label=f'Componente $x^r$ ({distancia_x_robot:.2f}m)')
    
    # Línea Componente Y (Desde la esquina hasta el Pato)
    ax.plot([corner_x, p_world[0]], [corner_y, p_world[1]], 
            color='darkred', linestyle=':', linewidth=2, label='Componente $y^r$')
    
    # Dibujar un punto en la "esquina" para que se entienda el ángulo recto
    ax.scatter(corner_x, corner_y, color='red', marker='x', s=50)

    # 5. El Objeto (Pato)
    ax.scatter(p_world[0], p_world[1], c='gold', s=180, edgecolors='black', label='Pato', zorder=10)
    
    plt.title(f"Descomposición en el Marco del Robot\nRotación {np.degrees(theta):.0f}°", fontsize=14)
    plt.legend(loc='lower right')
    plt.show()


def visualizar_transformacion_SE2(pose_robot, p_local):
    """
    Visualiza la transformación completa SE(2): Traslación + Rotación.
    pose_robot: [x, y, theta]
    p_local: [x, y, 1] (Coordenada homogénea local)
    """
    x_rob, y_rob, theta = pose_robot
    
    # Calcular la posición global del objeto (Matemática interna para graficar)
    c, s = np.cos(theta), np.sin(theta)
    # Matriz de Transformación
    T = np.array([
        [c, -s, x_rob],
        [s,  c, y_rob],
        [0,  0, 1]
    ])
    p_global = T @ p_local
    
    # --- INICIO DEL PLOT ---
    fig, ax = plt.subplots(figsize=(8, 8))
    limit = max(x_rob, p_global[0]) + 1.5
    ax.set_aspect('equal')
    ax.set_xlim(-1, limit)
    ax.set_ylim(-1, limit)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 1. Marco del MUNDO (Origen, Azul)
    ax.arrow(0, 0, 1, 0, head_width=0.1, color='blue', alpha=0.6, width=0.015)
    ax.arrow(0, 0, 0, 1, head_width=0.1, color='blue', alpha=0.6, width=0.015)
    ax.text(0, -0.3, "Mundo (0,0)", color='blue', fontweight='bold')
    
    # 2. Marco del ROBOT (Trasladado y Rotado, Rojo)
    # Ejes visuales del robot
    len_axis = 1.0
    ax.arrow(x_rob, y_rob, c*len_axis, s*len_axis, head_width=0.1, color='darkred', width=0.015) # Eje X robot
    ax.arrow(x_rob, y_rob, -s*len_axis, c*len_axis, head_width=0.1, color='darkred', width=0.015) # Eje Y robot
    ax.text(x_rob - 0.2, y_rob + 0.2, "Robot", color='darkred', fontweight='bold')
    
    # 3. VECTORES DE CONEXIÓN (La explicación visual)
    # A. Vector Traslación (Mundo -> Robot)
    ax.plot([0, x_rob], [0, y_rob], 'k--', alpha=0.4, label='Traslación Robot')
    
    # B. Vector Local Rotado (Robot -> Pato)
    ax.plot([x_rob, p_global[0]], [y_rob, p_global[1]], 'r--', linewidth=2, label='Visión Local (Rotada)')
    
    # 4. El Objeto (Pato)
    ax.scatter(p_global[0], p_global[1], c='gold', s=200, edgecolors='black', label='Objeto (Pato)', zorder=10)
    
    # Etiquetas de coordenadas
    ax.text(p_global[0]+0.1, p_global[1], f"Global\n({p_global[0]:.1f}, {p_global[1]:.1f})", fontsize=9, backgroundcolor='white')
    
    plt.title(f"Transformación SE(2) Completa\nRobot en ({x_rob},{y_rob}) rotado {np.degrees(theta):.0f}°", fontsize=14)
    plt.legend(loc='lower right')
    plt.show()


def visualizar_inversa(pose_robot, p_global):
    """
    Muestra la relación inversa: Dado un objetivo global, ¿dónde está para el robot?
    """
    x_rob, y_rob, theta = pose_robot
    
    # 1. Calcular T y T_inv
    c, s = np.cos(theta), np.sin(theta)
    T = np.array([[c, -s, x_rob], [s, c, y_rob], [0, 0, 1]])
    T_inv = np.linalg.inv(T)
    
    # 2. Calcular punto local (Matemática inversa)
    p_local = T_inv @ p_global
    
    # --- PLOT ---
    fig, ax = plt.subplots(figsize=(8, 8))
    limit = max(x_rob, p_global[0]) + 1.5
    ax.set_aspect('equal')
    ax.set_xlim(-1, limit)
    ax.set_ylim(-1, limit)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Marcos
    # Mundo
    ax.arrow(0, 0, 1, 0, head_width=0.1, color='blue', alpha=0.5)
    ax.arrow(0, 0, 0, 1, head_width=0.1, color='blue', alpha=0.5)
    ax.text(0.1, -0.3, "Origen Mundo", color='blue')
    
    # Robot
    ax.arrow(x_rob, y_rob, c, s, head_width=0.1, color='darkred', width=0.02)
    ax.text(x_rob, y_rob+0.3, "Robot", color='darkred', fontweight='bold')
    
    # Objetivo (Goal)
    ax.scatter(p_global[0], p_global[1], c='green', s=200, marker='*', edgecolors='black', label='Meta (Global)', zorder=10)
    ax.text(p_global[0]+0.2, p_global[1], f"Meta\n({p_global[0]},{p_global[1]})", color='green')

    # VECTOR DE NAVEGACIÓN (Del Robot a la Meta)
    # Dibujamos la línea de visión directa
    ax.plot([x_rob, p_global[0]], [y_rob, p_global[1]], 'g--', linewidth=1.5, label='Distancia a Meta')
    
    # Visualizar las componentes locales (Lo que el robot "siente")
    # Para graficar esto rotado, usamos trucos geométricos, pero conceptualmente:
    # Mostramos texto con lo que calculó la inversa
    info_text = (
        f"CÁLCULO INVERSO (Navegación):\n"
        f"Para llegar, el robot debe moverse:\n"
        f"x_local (Adelante): {p_local[0]:.2f} m\n"
        f"y_local (Lado): {p_local[1]:.2f} m"
    )
    plt.text(-0.5, limit-1.5, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title(f"Navegación: De Global a Local ($T^{{-1}}$)", fontsize=14)
    plt.legend(loc='lower right')
    plt.show()