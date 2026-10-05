import fdrl26_rl as rl

import numpy as np
import matplotlib.pyplot as plt

import imageio
from IPython.display import Image
from IPython.display import clear_output


def policy_symbolic_to_deterministic(policy_symbolic):
    codes = {"L": 0, "←": 0, 
             "D": 1, "↓": 1,
             "R": 2, "→": 2,
             "U": 3, "↑": 3,
             ".": 0, "0": 0}
    rows = [row.upper().split() for row in policy_symbolic]
    n = len(rows)

    if n == 0 or any(len(row) != n for row in rows):
        raise ValueError("Політика має бути квадратною таблицею N × N.")

    unknown = {symbol for row in rows for symbol in row} - codes.keys()
    if unknown:
        raise ValueError(f"Невідомі позначення дій: {sorted(unknown)}")

    return np.array([codes[symbol] for row in rows for symbol in row], dtype=int)


def plot_deterministic_policy(policy, map_layout, title="Детермінована політика"):
    """
    Візуалізує детерміновану політику на карті зі стрілками
    Visualizes deterministic policy on map with arrows
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    rows = len(map_layout)
    cols = len(map_layout[0])
    
    # Створюємо графік
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Відображаємо карту
    for i in range(rows):
        for j in range(cols):
            state = i * cols + j
            symbol = map_layout[i][j]
            
            # Кольори для різних типів клітинок
            if symbol == 'S':
                color = '#90EE90'  # Світло-зелений для старту
            elif symbol == 'G':
                color = '#FFD700'  # Золотий для цілі
            elif symbol == 'H':
                color = '#FFB6C1'  # Світло-червоний для прірв
            else:
                color = '#ADD8E6'  # Світло-блакитний для льоду
            
            # Малюємо клітинку
            rect = plt.Rectangle((j, rows-1-i), 1, 1, 
                               facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Додаємо символ
            if symbol in ['H', 'G']:
                ax.text(j + 0.5, rows-1-i + 0.5, symbol, 
                   fontsize=11, ha='center', va='center', fontweight='bold')
            
            # Додаємо стрілку політики (якщо не термінальний стан)
            if symbol not in ['H', 'G']:
                action = policy[state]
                arrow_dict = {
                    0: '←',  # LEFT
                    1: '↓',  # DOWN  
                    2: '→',  # RIGHT
                    3: '↑'   # UP
                }
                arrow = arrow_dict.get(action, '?')
                ax.text(j + 0.5, rows-1-i + 0.45, arrow, 
                       fontsize=21, ha='center', va='center', 
                       color='red', fontweight='bold')
    
    # Налаштування графіка
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels(np.arange(cols))
    ax.set_yticklabels(np.arange(rows)[::-1])
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()



def plot_img(pic, step, state, action, next_state, reward):
    action_symbols = {-1: '?', 0: '←', 1: '↓', 2: '→', 3: '↑'}
    plt.imshow(pic) 
    if step==-1:
        plt.title(f"Init State. $S_0={state}$")
    else:
        plt.title(f"$t={step+1}$. $S_{step}={state}$, $A_{step}={action}$({action_symbols[action]}), $S_{step+1}={next_state}$, $R_{step+1}={reward}$")
    plt.axis('off') # Відключення осей координат для чистого відображения зображення
    plt.show()




def print_trajectory(trajectory):
    actions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    print(" t |  s | дія | винагорода | s' | завершення")
    print("-" * 55)

    for t, (state, action, reward, next_state, terminated, truncated) in enumerate(trajectory):
        ending = "термінальний стан" if terminated else "ліміт кроків" if truncated else ""
        print(f"{t:>2} | {state:>2} |  {actions[action]}  | {reward:>10g} | {next_state:>2} | {ending}")


def print_result(trajectory):
    if not trajectory:
        print("Траєкторія порожня.")
        return

    _, _, last_reward, last_state, terminated, truncated = trajectory[-1]

    if terminated:
        outcome = "успіх (досягнуто цілі)" if last_reward == 1 else "невдача (ополонка)"
    elif truncated:
        outcome = "зупинено через ліміт кроків"
    else:
        outcome = "траєкторія не завершена"

    print(f"Кількість кроків: {len(trajectory)}")
    print(f"Останній стан: {last_state}")
    print(f"Результат: {outcome}")
    print(f"Сума винагород: {sum(step[2] for step in trajectory):g}")



def test_agent(env, policy, n_episodes=1):
    frames = []
    for i in range(n_episodes):
        state, info = env.reset()
        done = False
        truncated = False
        pic=env.render()
        #h, w, _ = pic.shape
        #print (i, h, w)
        frames.append(pic)
        while not (done or truncated):
            action = rl.agent(policy, state) 
            next_state, reward, done, truncated, extra_info = env.step(action)
            pic=env.render()
            frames.append(pic)
            state = next_state
    return frames

def play_and_save_gif(frames, path="video.gif", fps=2):
    #import imageio
    #from IPython.display import Image
    imageio.mimsave(path, frames, fps=fps, loop=0)
    return Image(path)    


def print_value_map(V, map_layout):
    """
    Compact output of value on map
    """
    rows = len(map_layout)
    cols = len(map_layout[0])

    print("=" * cols * 9)
    
    for i in range(rows):
        # Рядок з значеннями
        values_row = ""
        # Рядок з символами
        symbols_row = ""
        
        for j in range(cols):
            state_index = i * cols + j
            value = V[state_index]
            symbol = map_layout[i][j]
            
            values_row += f"{value:7.5f}  "
            symbols_row += f"   {symbol}     "
        
        print(values_row)
        print(symbols_row)
        print()
    
    print("=" * cols * 9)
