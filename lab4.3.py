"""ЛР 4.3. Модель автоматичного заходу за КРМ, моделювання бічного руху.
Python 3: numpy, matplotlib. Запуск: python lab_4_3_for_review.py
2D показує горизонтальну траєкторію. 3D додає геометричну модель:
H=min(H0, (D+300)*tan(3°)); окремо обчислюється сигнал ГРП за математичною моделлю.
Вертикальна динаміка і реальне захоплення глісади не моделюються.
"""
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button, RadioButtons

DT=0.01; V=78.; D0=18000.; S_NOM=167.; T_KRP=.2
H0=500.0  # м: початкова висота 
GLIDE_ANGLE_DEG=3.0  # приклад у межах 2–4.5° 
GRM_OFFSET_M=300.0  # приклад у межах 200–450 м 
S_GRP=560.0         # номінальна крутизна , мкА/град
T_GRP=0.2           # у межах 0.1–0.3 с
GRP_LIMIT=250.0     # вітчизняна система, обмеження сигналу

def glide_reference(d_m):
    """Висота глісади за математичною моделлю: Hгл = (Dзпс + 300) tan(3°)."""
    return np.maximum(np.asarray(d_m, dtype=float) + GRM_OFFSET_M, 0.) * math.tan(math.radians(GLIDE_ANGLE_DEG))

def geometric_height(d_m):
    """Лише для візуалізації: H0 до перетину, потім геометрична глісада."""
    return np.minimum(H0, glide_reference(d_m))

def grp_signal(dense):
    """Розрахунок ГРМ–ГРП  для умовної висоти.
    Повертає [t, H, Hгл, epsilon*_гл, I*_гл, I_гл, epsilon_гл].
    НЕ є моделлю поздовжньої динаміки чи замкненого керування висотою.
    """
    t = dense[:, 0]
    d = dense[:, 5]
    h = geometric_height(d)
    href = glide_reference(d)
    range_m = np.maximum(d + GRM_OFFSET_M, 1.0)
    # ε* = atan((H-Hгл)/D) — еквівалентність у моделі наближена
    angle = np.degrees(np.arctan((h - href)/range_m))
    raw = S_GRP * angle
    limited = np.clip(raw, -GRP_LIMIT, GRP_LIMIT)
    filtered = np.empty_like(limited)
    filtered[0] = limited[0]  # початковий стан фільтра — припущення
    for i in range(1, len(t)):
        step = t[i] - t[i-1]
        filtered[i] = filtered[i-1] + step * (limited[i-1]-filtered[i-1])/T_GRP
    return np.column_stack((t, h, href, angle, raw, filtered, filtered/S_GRP))

OUT=Path(__file__).resolve().parent/'lab43_results_for_review'; OUT.mkdir(exist_ok=True)
# Методичка не визначає початкових станів фільтрів.
# 'steady' встановлює початковий струм КРП рівним статичному сигналу;
# для ланки T15 початковий вихід нульовий. Це припущення, прийняте припущення.
FILTER_INIT='steady'
# Методичка передбачає зміну k5/k17 після захоплення глісади,
# але подію захоплення не визначено: використано значення ДО захоплення.
# Не слід називати цю реалізацію повною моделлю захоплення глісади.

def sat(x,limit): return max(-limit,min(limit,x))
def coeff():
    S,l,m,Ix,Iy,rho=201.45,37.55,80000.,250000.,900000.,.119
    q=S*l*rho*V**2/2; r=S*l*l*rho*V/4; h=S*rho*V/2
    # Таблиця 2, стор. 4; індекси і перехресні похідні збережено
    return dict(a1=-(-.21)*r/Iy,a2=-(-.2)*q/Iy,a3=-(-.0716)*q/Iy,
                a4=-(-1.0715)*h/m,a5=-(-.0206)*q/Ix,
                a6=-(-.31)*r/Ix,a7=-(-.183)*h/m,
                b1=-(-.583)*r/Ix,b2=-(-.186)*q/Ix,b3=-(-.0688)*q/Ix,
                b4=9.81/V*math.cos(math.radians(7.1)),b5=-(0.)*q/Iy,
                b6=-(-.006)*r/Iy,b7=math.sin(math.radians(7.1)))
C=coeff()
@dataclass(frozen=True)
class Case:
    name:str; z0:float; psi0:float; slope:float=167.
CASES=[Case('3.1_Z-5000',-5000,90),Case('3.1_Z-3000',-3000,90),Case('3.1_Z-2000',-2000,90),
       Case('3.2_S54',-3000,90,54),Case('3.2_S167',-3000,90,167),Case('3.2_S280',-3000,90,280),
       Case('3.3_Z300',300,0),Case('3.3_Z2000',2000,0)]
# y: psi_dot, gamma_dot, beta, psi, gamma, Z, D, I_KRP,
# LP_omega_x, LP_omega_y, LP_delta_psi(T5), LP_eps(T17), LP_after_F2(T15)
# У T*p/(T*p+1) стан LP'= (x-LP)/T, вихід x-LP.
def run(case,max_time=900.):
    y=np.zeros(13);y[3]=-case.psi0;y[5]=case.z0;y[6]=D0
    initial_angle=57.3*math.atan(case.z0/(D0+4000.))
    initial_current=sat(case.slope*initial_angle,250.)
    if FILTER_INIT=='steady':
        y[7]=initial_current
        y[10]=case.psi0
        y[11]=initial_current/S_NOM
        # Для початкових швидкостей обертання wx=wy=0 фільтри вже узгоджені.
    rows=[]; dense=[]; status='часовий ліміт'
    for n in range(int(max_time/DT)+1):
        t=n*DT
        wy,wx,beta,psi,gamma,z,d,ik,lwx,lwy,lpsi,leps,gcmd=y
        heading=-psi; delta_psi=heading
        angle=57.3*math.atan(z/(d+4000.))
        i_star=sat(case.slope*angle,250.)
        eps=ik/S_NOM
        f1=sat(-1.3*delta_psi+8.*eps,25.)
        # k5=2, k17=170 до захоплення глісади
        # Стор. 5: F1 входить у суму; F2 обмежує всю суму; T15 фільтрує вихід F2.
        outer=1.3*delta_psi+(2./2.3)*(delta_psi-lpsi)+(170./2.3)*(eps-leps)+f1
        gamma_command=-gcmd
        de=sat(2.*(gamma-gamma_command)+1.5*(wx-lwx),12.)
        dn=sat(2.5*(wy-lwy)+2.*wy,10.)
        dwy=-C['a1']*wy-C['b6']*wx-C['a2']*beta-C['a3']*dn-C['b5']*de
        dwx=-C['a6']*wy-C['b1']*wx-C['b2']*beta-C['a5']*dn-C['b3']*de
        dbeta=wy+C['b7']*wx+C['b4']*gamma-C['a4']*beta-C['a7']*dn
        dz=V*math.sin(math.radians(delta_psi+beta))
        dd=-V*math.cos(math.radians(delta_psi+beta))
        record=[t,de,dn,gamma,heading,d,z,ik,angle,gamma_command]
        dense.append(record)
        if n%500==0: rows.append(record)
        if d<=0:
            if n%500: rows.append(record)
            status='торець ЗПС';break
        dy=np.array([dwy,dwx,dbeta,wy,wx,dz,dd,(i_star-ik)/T_KRP,
                     (wx-lwx)/1.6,(wy-lwy)/2.5,(delta_psi-lpsi)/2.3,
                     (eps-leps)/2.3,(sat(outer,20.)-gcmd)/.85])
        y+=DT*dy
        if not np.all(np.isfinite(y)) or max(abs(y[0]),abs(y[1]),abs(y[2]),abs(y[3]),abs(y[4]))>1e5:
            status='нестійкість/розбіжність';break
    return np.array(rows),status,np.array(dense)

def metrics(dense, case):
    """Показники за повним рядом (0,01 с).

    Початок захоплення КРМ — операційне визначення: перший момент,
    коли необмежений розрахунковий струм потрапляє в межі ±250 мкА.
    Це НЕ окремо задана моделлю подія захоплення глісади.
    """
    t = dense[:, 0]
    i = dense[:, 7]
    angle = dense[:, 8]
    raw = case.slope * angle
    inside = np.flatnonzero(np.abs(raw) < 250.0 - 1e-8)
    capture_idx = int(inside[0]) if len(inside) else None
    capture_time = float(t[capture_idx]) if capture_idx is not None else float('nan')

    # Перший перетин нуля після входження у ненасичену зону.
    crossing_idx = None
    if capture_idx is not None:
        sign0 = np.sign(raw[capture_idx])
        if sign0:
            candidates = np.flatnonzero(sign0 * i[capture_idx:] <= 0)
            if len(candidates):
                crossing_idx = capture_idx + int(candidates[0])
    if crossing_idx is None:
        overshoot = float('nan')
        crossing_time = float('nan')
    else:
        crossing_time = float(t[crossing_idx])
        sign0 = np.sign(raw[capture_idx])
        overshoot = float(np.max(np.maximum(0, -sign0 * i[crossing_idx:])))

    # Останній вихід за межі ±35 мкА; відлік після входу в зону.
    settle_time = float('nan')
    if capture_idx is not None:
        # Критерій остаточного входження після початку захоплення зони.
        violations = np.flatnonzero(np.abs(i[capture_idx:]) > 35.0)
        if len(violations) == 0:
            settle_time = float(t[capture_idx])
        else:
            last = capture_idx + int(violations[-1])
            if last + 1 < len(t):
                settle_time = float(t[last + 1])

    return {'capture_proxy_time_s': capture_time,
            'first_zero_crossing_s': crossing_time,
            'settle_time_s': settle_time,
            'overshoot_uA': overshoot,
            'overshoot_le_42': ('так' if np.isfinite(overshoot) and overshoot <= 42 else
                                 'ні' if np.isfinite(overshoot) else 'не визначено'),
            'max_abs_aileron_deg': float(np.max(np.abs(dense[:, 1]))),
            'max_abs_rudder_deg': float(np.max(np.abs(dense[:, 2]))),
            'max_abs_roll_deg': float(np.max(np.abs(dense[:, 3]))),
            'max_abs_KRP_uA': float(np.max(np.abs(i)))}

def output(results):
    print('\nЛАБОРАТОРНА РОБОТА № 4.3 — РОЗРАХУНКИ ЗА МЕТОДИЧКОЮ')
    print('Коефіцієнти диференціальних рівнянь (стор. 4):')
    for k,v in C.items():print(f'  {k:>3} = {v:.9g}')
    print('Умови: Ейлер, крок 0,01 с; друк кожні 5 с; без завад.')
    print('Початкові стани фільтрів: steady (прийняте припущення).')
    with open(OUT/'coefficients.txt','w',encoding='utf-8') as f:
        for k,v in C.items():f.write(f'{k} = {v:.10g}\n')
    header='t_s,aileron_deg,rudder_deg,roll_deg,gyro_heading_deg,D_m,Z_m,KRP_uA,beacon_angle_deg,command_roll_deg'
    groups={'3.1':['3.1_Z-5000','3.1_Z-3000','3.1_Z-2000'],
            '3.2':['3.2_S54','3.2_S167','3.2_S280'],
            '3.3':['3.3_Z300','3.3_Z2000']}
    summary=[]
    for name,(a,status,dense) in results.items():
        coords_3d = np.column_stack((a[:, 0], a[:, 5], a[:, 6], geometric_height(a[:, 5])))
        np.savetxt(OUT/f'{name}_3d_geometric.csv', coords_3d, delimiter=',',
                   header='t_s,D_m,Z_m,H_geometric_m', comments='', fmt='%.5f')
        np.savetxt(OUT/f'{name}.csv',a,delimiter=',',header=header,comments='',fmt='%.5f')
        print('\n' + '='*106)
        print(f'ДОСЛІД {name} | S={next(c.slope for c in CASES if c.name == name):g} мкА/град | завершення: {status}')
        print('='*106)
        print(f'{"t, с":>7} {"Елер., °":>10} {"Руль, °":>10} {"Крен, °":>10} {"Курс, °":>10} {"D, м":>12} {"Z, м":>12} {"КРП, мкА":>12}')
        print('-'*106)
        for r in a:
            print(f'{r[0]:7.2f} {r[1]:10.3f} {r[2]:10.3f} {r[3]:10.3f} {r[4]:10.3f} {r[5]:12.2f} {r[6]:12.2f} {r[7]:12.3f}')
        print(f'Кінець моделювання: t={a[-1,0]:.2f} с, D={a[-1,5]:.2f} м, Z={a[-1,6]:.2f} м')
        case = next(c for c in CASES if c.name == name)
        summary.append([name,status,a[-1,0],a[-1,5],a[-1,6],a[-1,7],*metrics(dense,case).values()])
        fig,axs=plt.subplots(3,2,figsize=(13,10),sharex=True)
        specs=[(6,'Бічне відхилення Z, м'),(4,'Гіроскопічний курс, град'),
               (3,'Кут крену, град'),(7,'Струм КРП, мкА'),
               (1,'Елерони, град'),(2,'Руль напрямку, град')]
        for ax,(col,ylab) in zip(axs.flat,specs):
            ax.plot(dense[:,0],dense[:,col],lw=1.5)
            ax.set_ylabel(ylab);ax.grid(alpha=.35)
            if col==7:
                for level in (-35,35):ax.axhline(level,ls='--',color='gray',lw=1)
                # ±42 мкА — критерій перерегулювання, не обмеження миттєвого струму.
        for ax in axs[-1,:]:ax.set_xlabel('Час, с')
        fig.suptitle(name+' — '+status);fig.tight_layout()
        fig.savefig(OUT/f'parameters_{name}.png',dpi=140);plt.close(fig)
    with open(OUT/'summary.csv','w',encoding='utf-8') as f:
        f.write('experiment,status,time_s,D_m,Z_m,final_KRP_uA,capture_proxy_time_s,first_zero_crossing_s,settle_time_s,overshoot_uA,overshoot_le_42,max_abs_aileron_deg,max_abs_rudder_deg,max_abs_roll_deg,max_abs_KRP_uA\n')
        for row in summary:
            f.write(','.join(str(v) for v in row)+'\n')
    for group,names in groups.items():
        fig,ax=plt.subplots(figsize=(11,6))
        for name in names:
            a=results[name][0];ax.plot(a[:,5]/1000,a[:,6]/1000,label=name)
        ax.axhline(0,ls='--',color='black');ax.axvspan(-3,0,alpha=.15,color='gray')
        mark_beacons_2d(ax)
        ax.invert_xaxis();ax.set(xlabel='D, км',ylabel='Z, км',title='Дослід '+group)
        ax.grid();ax.legend();fig.tight_layout();fig.savefig(OUT/f'trajectories_{group}.png',dpi=160);plt.close(fig)

# Розташування радіомаяків у системі координат графіків (км).
# КРМ: 3000 м ЗПС + 1000 м за дальнім торцем.
# ГРМ: 300 м від ближнього торця та 150 м збоку від осі.
KRM_X_KM, KRM_Z_KM = -4.0, 0.0
GRM_X_KM, GRM_Z_KM = -0.3, 0.15

def mark_beacons_2d(ax):
    ax.scatter([KRM_X_KM], [KRM_Z_KM], marker='^', s=95,
               color='darkorange', edgecolor='black', linewidth=.5,
               label='КРМ', zorder=9)
    ax.scatter([GRM_X_KM], [GRM_Z_KM], marker='s', s=75,
               color='mediumorchid', edgecolor='black', linewidth=.5,
               label='ГРМ', zorder=9)
    ax.annotate('КРМ', (KRM_X_KM, KRM_Z_KM), xytext=(-6, 11),
                textcoords='offset points', fontsize=9, weight='bold')
    ax.annotate('ГРМ', (GRM_X_KM, GRM_Z_KM), xytext=(7, 10),
                textcoords='offset points', fontsize=9, weight='bold')

def mark_beacons_3d(ax):
    ax.scatter([KRM_X_KM], [KRM_Z_KM], [0], marker='^', s=95,
               color='darkorange', depthshade=False, label='КРМ')
    ax.scatter([GRM_X_KM], [GRM_Z_KM], [0], marker='s', s=75,
               color='mediumorchid', depthshade=False, label='ГРМ')
    ax.text(KRM_X_KM, KRM_Z_KM, .045, 'КРМ', fontsize=9)
    ax.text(GRM_X_KM, GRM_Z_KM, .045, 'ГРМ', fontsize=9)

def animate(results, initial='3.1_Z-3000'):
    """Інтерактивний перегляд усіх 8 траєкторій із перемиканням дослідів."""
    names = list(results)
    if initial not in results:
        initial = names[0]
    data = {'a': results[initial][0], 'name': initial}
    fig, ax = plt.subplots(figsize=(12.8, 7))
    fig.subplots_adjust(left=.09, bottom=.23, right=.77, top=.91)
    ax.axhline(0, color='black', ls='--', label='Вісь ЗПС')
    ax.add_patch(Rectangle((-3, -.05), 3, .1, color='gray', alpha=.5, label='ЗПС'))
    ax.scatter([-4], [0], marker='^', color='orange', s=100, label='КРМ')
    path, = ax.plot([], [], color='tab:blue', label='Траєкторія')
    point, = ax.plot([], [], 'ro', markersize=9)
    arrow, = ax.plot([], [], 'r-', lw=2)
    info = ax.text(.02, .98, '', transform=ax.transAxes, va='top',
                   bbox={'facecolor': 'white', 'alpha': .9})
    ax.set_xlabel('D, км')
    ax.set_ylabel('Z, км')
    ax.grid()
    ax.legend(loc='upper right')

    slider_ax = fig.add_axes([.14, .125, .60, .035])
    time_slider = Slider(slider_ax, 'Час, с', 0, float(data['a'][-1, 0]),
                         valinit=0, valfmt='%.1f')
    play_ax = fig.add_axes([.14, .045, .14, .055])
    back_ax = fig.add_axes([.30, .045, .09, .055])
    next_ax = fig.add_axes([.41, .045, .09, .055])
    reset_ax = fig.add_axes([.52, .045, .12, .055])
    speed_ax = fig.add_axes([.69, .05, .12, .035])
    play_button = Button(play_ax, 'Відтворити')
    back_button = Button(back_ax, '◀ 5 с')
    next_button = Button(next_ax, '5 с ▶')
    reset_button = Button(reset_ax, 'Спочатку')
    speed_slider = Slider(speed_ax, 'Швидк.', .25, 4., valinit=1., valstep=.25)

    radio_ax = fig.add_axes([.795, .36, .19, .43])
    radio_ax.set_title('Оберіть дослід', fontsize=11)
    radios = RadioButtons(radio_ax, names, active=names.index(initial))
    state = {'index': 0, 'playing': False, 'setting_slider': False, 'fraction': 0.}

    def draw(i, sync_slider=True):
        a = data['a']
        i = int(np.clip(i, 0, len(a)-1))
        state['index'] = i
        r = a[i]
        x, z = r[5]/1000, r[6]/1000
        heading = math.radians(r[4])
        length = .5
        point.set_data([x], [z])
        arrow.set_data([x, x-length*math.cos(heading)],
                       [z, z+length*math.sin(heading)])
        info.set_text(f't={r[0]:.1f} с\nD={r[5]:.0f} м\nZ={r[6]:.0f} м\n'
                      f'Курс={r[4]:.1f}°\nКРП={r[7]:.1f} мкА')
        if sync_slider:
            state['setting_slider'] = True
            time_slider.set_val(float(r[0]))
            state['setting_slider'] = False
        fig.canvas.draw_idle()
        return point, arrow, info

    def select_case(name):
        state['playing'] = False
        state['fraction'] = 0.
        play_button.label.set_text('Відтворити')
        a = results[name][0]
        data['a'], data['name'] = a, name
        path.set_data(a[:, 5]/1000, a[:, 6]/1000)
        ax.set_xlim(-4, max(19, float(np.max(a[:, 5])/1000)+1))
        pad = max(1., float(np.max(np.abs(a[:, 6]))/1000)*1.2)
        ax.set_ylim(-pad, pad)
        ax.invert_xaxis()
        ax.set_title('2D автоматичний захід: ' + name)
        time_slider.valmax = float(a[-1, 0])
        time_slider.ax.set_xlim(0, float(a[-1, 0]))
        draw(0)

    def seek(value):
        if state['setting_slider']:
            return
        times = data['a'][:, 0]
        i = int(np.argmin(np.abs(times-float(value))))
        state['fraction'] = 0.
        draw(i, sync_slider=False)

    def toggle(event):
        a = data['a']
        if state['index'] >= len(a)-1 and not state['playing']:
            draw(0)
        state['playing'] = not state['playing']
        play_button.label.set_text('Пауза' if state['playing'] else 'Відтворити')
        fig.canvas.draw_idle()

    def step(seconds):
        state['playing'] = False
        play_button.label.set_text('Відтворити')
        times = data['a'][:, 0]
        target = np.clip(times[state['index']]+seconds, times[0], times[-1])
        draw(int(np.argmin(np.abs(times-target))))

    def restart(event):
        state['playing'] = False
        play_button.label.set_text('Відтворити')
        draw(0)

    time_slider.on_changed(seek)
    play_button.on_clicked(toggle)
    back_button.on_clicked(lambda event: step(-5))
    next_button.on_clicked(lambda event: step(5))
    reset_button.on_clicked(restart)
    radios.on_clicked(select_case)

    def tick(_):
        a = data['a']
        if state['playing'] and state['index'] < len(a)-1:
            state['fraction'] += float(speed_slider.val)
            advance = int(state['fraction'])
            state['fraction'] -= advance
            if advance:
                draw(min(state['index']+advance, len(a)-1))
        elif state['playing'] and state['index'] >= len(a)-1:
            state['playing'] = False
            play_button.label.set_text('Відтворити')
        return point, arrow, info

    select_case(initial)
    animation = FuncAnimation(fig, tick, frames=None, interval=90,
                              repeat=False, cache_frame_data=False)
    fig._lab43_widgets = (time_slider, play_button, back_button,
                          next_button, reset_button, speed_slider, radios)
    fig._lab43_animation = animation
    fig._lab43_callbacks = (seek, toggle, step, restart, tick, select_case)
    return animation


def view_3d(results, initial='3.1_Z-3000'):
    """3D: бічна траєкторія з рівнянь, H — геометричний профіль.

    До D_capture=H0/tan(3°) висота H0; після H=D*tan(3°).
    Це не вертикальна динаміка і не алгоритм захоплення глісади.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    names = list(results)
    fig = plt.figure(figsize=(13.5, 8))
    ax = fig.add_axes([.07, .22, .68, .70], projection='3d')
    fig.text(.07, .95, '3D: H=500 м до геометричного перетину з глісадою 3°; вертикальна САУ відсутня',
             fontsize=11, color='darkred')
    radio_ax = fig.add_axes([.78, .40, .20, .46])
    radio_ax.set_title('Дослід')
    radios = RadioButtons(radio_ax, names, active=names.index(initial))
    time_ax = fig.add_axes([.17, .125, .56, .035])
    time_slider = Slider(time_ax, 'Час, с', 0, float(results[initial][0][-1, 0]), valinit=0)
    play_ax = fig.add_axes([.16, .045, .14, .055])
    reset_ax = fig.add_axes([.32, .045, .13, .055])
    speed_ax = fig.add_axes([.57, .055, .16, .03])
    play_button = Button(play_ax, 'Відтворити')
    reset_button = Button(reset_ax, 'Спочатку')
    speed_slider = Slider(speed_ax, 'Швидк.', .25, 5, valinit=1, valstep=.25)
    state = {'name': initial, 'index': 0, 'playing': False, 'fraction': 0., 'setting': False}
    objects = {}
    slope = math.tan(math.radians(GLIDE_ANGLE_DEG))
    capture_d_km = H0 / slope / 1000.

    def select(name):
        state.update(name=name, index=0, playing=False, fraction=0.)
        play_button.label.set_text('Відтворити')
        a = results[name][0]
        d = a[:, 5]/1000
        z = a[:, 6]/1000
        h = geometric_height(a[:, 5]) / 1000.
        ax.cla()
        ax.plot([18, 0], [0, 0], [18*slope, 0], '--', color='orange',
                label='Глісада 3° (геометричний орієнтир)')
        ax.scatter([capture_d_km], [0], [H0/1000], marker='x',
                   color='darkred', s=70, label='Перетин H=500 м із глісадою')
        ax.plot([18, 0], [0, 0], [0, 0], ':', color='gray', label='Вісь ЗПС')
        ax.plot([-0.5, 0], [0, 0], [0, 0], lw=7, color='dimgray', label='ЗПС')
        line, = ax.plot(d, z, h, color='tab:blue', lw=2, label='Траєкторія (геометрична висота)')
        marker, = ax.plot([], [], [], 'ro', markersize=9, label='Літак')
        objects['marker'] = marker
        ax.set_xlabel('D, км')
        ax.set_ylabel('Z, км')
        ax.set_zlabel('H, км')
        ax.set_xlim(19, -1)
        side = max(1, np.max(np.abs(z))*1.15)
        ax.set_ylim(-side, side)
        ax.set_zlim(0, 1.05)
        ax.view_init(elev=24, azim=-67)
        ax.set_title('3D автоматичний захід: '+name+' (геометрична глісада)')
        ax.legend(loc='upper left', fontsize=8)
        time_slider.valmax = float(a[-1, 0])
        time_slider.ax.set_xlim(0, float(a[-1, 0]))
        draw(0)

    def draw(i, sync=True):
        a = results[state['name']][0]
        i = max(0, min(int(i), len(a)-1))
        state['index'] = i
        d, z = a[i, 5]/1000, a[i, 6]/1000
        h = float(geometric_height(a[i, 5])) / 1000.
        objects['marker'].set_data_3d([d], [z], [h])
        if sync:
            state['setting'] = True
            time_slider.set_val(float(a[i, 0]))
            state['setting'] = False
        fig.canvas.draw_idle()

    def seek(t):
        if not state['setting']:
            a = results[state['name']][0]
            draw(np.searchsorted(a[:, 0], t), sync=False)

    def toggle(event):
        a = results[state['name']][0]
        if state['index'] >= len(a)-1:
            draw(0)
        state['playing'] = not state['playing']
        play_button.label.set_text('Пауза' if state['playing'] else 'Відтворити')

    def restart(event):
        state['playing'] = False
        play_button.label.set_text('Відтворити')
        draw(0)

    def tick(_):
        if state['playing']:
            a = results[state['name']][0]
            state['fraction'] += speed_slider.val
            n = int(state['fraction'])
            state['fraction'] -= n
            if n:
                draw(min(len(a)-1, state['index']+n))
            if state['index'] >= len(a)-1:
                state['playing'] = False
                play_button.label.set_text('Відтворити')
        return (objects['marker'],)

    radios.on_clicked(select)
    time_slider.on_changed(seek)
    play_button.on_clicked(toggle)
    reset_button.on_clicked(restart)
    select(initial)
    animation = FuncAnimation(fig, tick, interval=90, cache_frame_data=False)
    fig._lab43_widgets = (radios, time_slider, play_button, reset_button, speed_slider)
    fig._lab43_callbacks = (select, draw, seek, toggle, restart, tick)
    fig._lab43_animation = animation
    return fig

def graph_viewer(results):
    """Окреме інтерактивне вікно для перегляду всіх результатів."""
    names = list(results)
    options = ['Траєкторії 3.1', 'Траєкторії 3.2', 'Траєкторії 3.3'] + names
    fig = plt.figure(figsize=(14, 8))
    fig.canvas.manager.set_window_title('ЛР 4.3 — графіки досліджень')
    fig.subplots_adjust(left=.25, right=.97, bottom=.09, top=.91, hspace=.45, wspace=.32)
    menu_ax = fig.add_axes([.015, .19, .21, .69])
    menu_ax.set_title('Оберіть дослід', fontsize=11)
    radios = RadioButtons(menu_ax, options, active=0)
    fig.text(.025, .08, 'Графіки збережено у папці\nlab43_results_for_review (PNG, CSV)', fontsize=9)

    def render(label):
        # Видаляємо попередні графіки, залишаючи панель вибору.
        for ax in list(fig.axes):
            if ax is not menu_ax:
                ax.remove()
        if label.startswith('Траєкторії'):
            group = label.split()[-1]
            group_names = [n for n in names if n.startswith(group + '_')]
            ax = fig.add_subplot(111)
            ax.set_position([.29, .15, .67, .71])
            for name in group_names:
                a = results[name][0]
                ax.plot(a[:,5]/1000, a[:,6]/1000, lw=2, label=name)
            ax.axhline(0, color='black', linestyle='--', label='Вісь ЗПС')
            ax.axvspan(-3, 0, color='gray', alpha=.16, label='ЗПС')
            mark_beacons_2d(ax)
            ax.invert_xaxis()
            ax.set(xlabel='Відстань до ЗПС D, км', ylabel='Бічне відхилення Z, км',
                   title='Порівняння траєкторій — дослід ' + group)
            ax.legend(loc='best', fontsize=9)
            ax.grid(alpha=.3)
        else:
            a, status, dense = results[label]
            plots = [(6, 'Бічне відхилення Z, м'), (4, 'Гіроскопічний курс, град'),
                     (3, 'Кут крену, град'), (7, 'Струм КРП, мкА'),
                     (1, 'Відхилення елеронів, град'), (2, 'Руль напрямку, град')]
            for i, (col, title) in enumerate(plots):
                ax = fig.add_subplot(3, 2, i+1)
                ax.set_position([.30+(i%2)*.34, .70-(i//2)*.29, .30, .20])
                ax.plot(dense[:,0], dense[:,col], lw=1.6)
                ax.set_title(title, fontsize=10)
                ax.set_xlabel('Час, с', fontsize=9)
                ax.grid(alpha=.3)
                if col == 7:
                    ax.axhline(35, ls='--', color='tab:green', lw=1, label='±35 мкА')
                    ax.axhline(-35, ls='--', color='tab:green', lw=1)
                    # Перерегулювання ±42 мкА оцінюється після переходу через нуль.
                    ax.legend(fontsize=8)
            fig.suptitle(f'Параметри польоту — {label} ({status})', x=.62, fontsize=13)
        if label.startswith('Траєкторії'):
            fig.suptitle('ЛР № 4.3 — графічні результати', x=.62, fontsize=13)
        fig.canvas.draw_idle()

    radios.on_clicked(render)
    fig._lab43_widgets = (radios,)
    fig._lab43_callbacks = (render,)
    render(options[0])
    return fig, radios



def compare_grouped(results, mode="2d"):
    """Окреме вікно 2D або 3D для синхронного порівняння траєкторій."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    groups = {
        '3.1 — бічне відхилення': ['3.1_Z-5000', '3.1_Z-3000', '3.1_Z-2000'],
        '3.2 — крутизна КРМ': ['3.2_S54', '3.2_S167', '3.2_S280'],
        '3.3 — паралельний курс': ['3.3_Z300', '3.3_Z2000'],
    }
    colors = ['tab:blue', 'tab:orange', 'tab:purple']
    fig = plt.figure(figsize=(13.5, 8.7))
    if mode == "2d":
        ax2 = fig.add_axes([.08, .46, .71, .41])
        ax3 = None
        axi = fig.add_axes([.08, .19, .71, .13])
    else:
        ax2 = None
        ax3 = fig.add_axes([.08, .38, .71, .49], projection='3d')
        axi = fig.add_axes([.08, .19, .71, .14])
    axradio = fig.add_axes([.82, .55, .17, .25])
    axradio.set_title('Порівняння')
    radio = RadioButtons(axradio, list(groups))
    ax_time = fig.add_axes([.13, .115, .68, .028])
    slider = Slider(ax_time, 'Час, с', 0, 255, valinit=0, valfmt='%.1f')
    ax_play = fig.add_axes([.14, .043, .13, .048])
    ax_back = fig.add_axes([.29, .043, .09, .048])
    ax_next = fig.add_axes([.40, .043, .09, .048])
    ax_reset = fig.add_axes([.51, .043, .12, .048])
    ax_speed = fig.add_axes([.72, .052, .12, .027])
    btn_play = Button(ax_play, 'Відтворити')
    btn_back = Button(ax_back, '◀ 5 с')
    btn_next = Button(ax_next, '5 с ▶')
    btn_reset = Button(ax_reset, 'Спочатку')
    speed = Slider(ax_speed, 'Швидк.', .25, 8., valinit=1., valstep=.25)
    state = {'group': list(groups)[0], 'time': 0., 'playing': False,
             'setting': False, 'artists': [], 'cache': []}
    fig.text(.055, .945, f'ЛР 4.3 — порівняння траєкторій ({mode.upper()})', fontsize=16, weight='bold')
    fig.text(.055, .910,
             '2D: бічний рух за моделлю' if mode == '2d' else
             '3D: бічний рух і сигнал ГРП; висота умовна',
             fontsize=10, color='firebrick')
    fig.text(.845, .44, 'Кольори однакові\nна всіх графіках.\n\nКерування часом\nспільне для всіх\nлітаків.', fontsize=10, va='top')

    def redraw():
        group = state['group']
        names = groups[group]
        ax2.clear() if ax2 is not None else None
        ax3.clear() if ax3 is not None else None
        axi.clear() if axi is not None else None
        if ax2 is not None:
            ax2.axhline(0, color='black', ls='--', lw=1, label='Вісь ЗПС')
            ax2.plot([0, -3], [0, 0], color='gray', lw=7, alpha=.6)
            mark_beacons_2d(ax2)
            ax2.set_xlabel('D, км'); ax2.set_ylabel('Z, км')
            ax2.set_title('Траєкторії зверху (2D)')
            ax2.grid(alpha=.3)
            axi.axhline(35, color='tab:green', ls='--', lw=1)
            axi.axhline(-35, color='tab:green', ls='--', lw=1, label='±35 мкА')
            axi.set_xlabel('Час, с'); axi.set_ylabel('КРП, мкА')
            axi.set_title('Сигнали курсового радіоприймача')
            axi.grid(alpha=.3)
        else:
            ax3.plot([18, 0], [0, 0],
                     [18*math.tan(math.radians(GLIDE_ANGLE_DEG)), 0],
                     ls='--', color='gray', lw=1.5, label='Глісада 3°')
            ax3.plot([0, -3], [0, 0], [0, 0], color='gray', lw=5)
            mark_beacons_3d(ax3)
            ax3.set_xlabel('D, км'); ax3.set_ylabel('Z, км'); ax3.set_zlabel('H, км')
            ax3.set_title('Просторові траєкторії (3D)')
            axi.axhline(0, color='black', ls='--', lw=1)
            axi.set_title('Сигнал ГРП за математичною моделлю (умовна висота)', fontsize=10)
            axi.set_ylabel('ГРП, мкА'); axi.set_xlabel('Час, с')
            axi.grid(alpha=.3)
        max_z = 1.; max_t = 1.
        state['artists'] = []
        state['cache'] = []
        for i, name in enumerate(names):
            a = results[name][2]  # dense: t, de, dn, gamma, heading, D, Z, I, ...
            step = max(1, len(a)//1200)
            coarse = a[::step]
            if coarse[-1, 0] != a[-1, 0]:
                coarse = np.vstack((coarse, a[-1]))
            x, z = coarse[:,5]/1000, coarse[:,6]/1000
            h = geometric_height(coarse[:,5])/1000
            color = colors[i]
            if ax2 is not None:
                ax2.plot(x, z, color=color, alpha=.22, lw=1.7)
                axi.plot(coarse[:,0], coarse[:,7], color=color, alpha=.22, lw=1.3)
                line, = ax2.plot([], [], color=color, lw=2.5, label=name)
                signal, = axi.plot([], [], color=color, lw=2)
                point, = ax2.plot([], [], 'o', color=color, markersize=9)
                signal_point, = axi.plot([], [], 'o', color=color, markersize=6)
                state['artists'].append((coarse,line,point,signal,signal_point))
            else:
                ax3.plot(x, z, h, color=color, alpha=.22, lw=1.5)
                line, = ax3.plot(x, z, h, color=color, lw=2.2, label=name)
                point, = ax3.plot([], [], [], 'o', color=color, markersize=8)
                grp = grp_signal(coarse)
                axi.plot(grp[:,0], grp[:,5], color=color, alpha=.22, lw=1.2)
                signal, = axi.plot([], [], color=color, lw=2)
                signal_point, = axi.plot([], [], 'o', color=color, markersize=6)
                state['artists'].append((coarse,line,point,signal,signal_point))
            state['cache'].append({'t': coarse[:,0], 'd': x, 'z': z, 'h': h,
                                   'i': coarse[:,7],
                                   'grp': grp[:,5] if ax3 is not None else None})
            max_z=max(max_z, float(np.max(np.abs(z))))
            max_t=max(max_t, float(a[-1,0]))
        if ax2 is not None:
            ax2.set_xlim(19, -4.7); ax2.set_ylim(-max_z*1.22, max_z*1.22)
            axi.set_xlim(0,max_t)
            ax2.legend(loc='upper right', fontsize=8)
        else:
            ax3.set_xlim(19, -4.7); ax3.set_ylim(-max_z*1.22, max_z*1.22)
            ax3.set_zlim(0, .65); ax3.view_init(elev=23, azim=-68)
            ax3.legend(loc='upper left', fontsize=8)
            axi.set_xlim(0,max_t)
        slider.valmax=max_t; slider.ax.set_xlim(0,max_t)
        state['time']=0.
        update(0.)

    def update(t, move_slider=True):
        state['time'] = float(t)
        for cache, (a,line,point,signal,signal_point) in zip(state['cache'], state['artists']):
            n=int(np.searchsorted(cache['t'], t, side='right'))
            n=max(1,min(n,len(cache['t'])))
            d=cache['d']; z=cache['z']
            if ax2 is not None:
                line.set_data(d[:n],z[:n])
                point.set_data([d[n-1]],[z[n-1]])
                signal.set_data(cache['t'][:n],cache['i'][:n])
                signal_point.set_data([cache['t'][n-1]],[cache['i'][n-1]])
            else:
                # У 3D статична траєкторія вже намальована; рухається тільки маркер.
                # Повторне промальовування всієї 3D-лінії суттєво сповільнює Matplotlib.
                point.set_data_3d([d[n-1]],[z[n-1]],[cache['h'][n-1]])
                signal.set_data(cache['t'][:n],cache['grp'][:n])
                signal_point.set_data([cache['t'][n-1]],[cache['grp'][n-1]])
        if move_slider:
            state['setting']=True
            slider.set_val(float(t))
            state['setting']=False
        fig.canvas.draw_idle()

    def choose(group):
        state['playing']=False
        btn_play.label.set_text('Відтворити')
        state['group']=group
        redraw()

    def seek(t):
        if not state['setting']:
            update(t, move_slider=False)

    def toggle(event):
        if state['time'] >= slider.valmax:
            update(0.)
        state['playing']=not state['playing']
        btn_play.label.set_text('Пауза' if state['playing'] else 'Відтворити')

    def shift(seconds):
        state['playing']=False
        btn_play.label.set_text('Відтворити')
        update(max(0., min(slider.valmax, state['time']+seconds)))

    def reset(event):
        state['playing']=False
        btn_play.label.set_text('Відтворити')
        update(0.)

    def tick(_):
        if state['playing']:
            t=min(slider.valmax, state['time']+float(speed.val)*.75)
            update(t)
            if t >= slider.valmax:
                state['playing']=False
                btn_play.label.set_text('Відтворити')
        return []

    radio.on_clicked(choose)
    slider.on_changed(seek)
    btn_play.on_clicked(toggle)
    btn_back.on_clicked(lambda event:shift(-5.))
    btn_next.on_clicked(lambda event:shift(5.))
    btn_reset.on_clicked(reset)
    redraw()
    anim=FuncAnimation(fig,tick,interval=110 if mode == '3d' else 65,cache_frame_data=False)
    fig._grouped_controls=(radio,slider,btn_play,btn_back,btn_next,btn_reset,speed)
    fig._grouped_callbacks=(choose,seek,toggle,shift,reset,tick,update)
    fig._grouped_animation=anim
    return fig



def export_group_gifs(results, frames=75, fps=10, only_groups=None):
    """GIF для Word: лише 2D траєкторії та синхронні сигнали КРП."""
    from matplotlib.animation import PillowWriter
    groups = {
        '3.1': ['3.1_Z-5000', '3.1_Z-3000', '3.1_Z-2000'],
        '3.2': ['3.2_S54', '3.2_S167', '3.2_S280'],
        '3.3': ['3.3_Z300', '3.3_Z2000'],
    }
    colors = ['tab:blue', 'tab:orange', 'tab:purple']
    for group, names in groups.items():
        if only_groups is not None and group not in only_groups:
            continue
        fig, (ax, axi) = plt.subplots(2, 1, figsize=(10, 6.2),
                                      gridspec_kw={'height_ratios': [3, 1], 'hspace': .48})
        fig.subplots_adjust(left=.09, right=.96, top=.93, bottom=.11)
        ax.axhline(0, color='black', ls='--', lw=1, label='Вісь ЗПС')
        ax.plot([0, -3], [0, 0], color='gray', lw=7, alpha=.6)
        mark_beacons_2d(ax)
        ax.set(xlabel='D, км', ylabel='Z, км', title='Траєкторії зверху (2D) — дослід '+group)
        ax.grid(alpha=.3)
        axi.axhline(35, color='tab:green', ls='--', lw=1)
        axi.axhline(-35, color='tab:green', ls='--', lw=1)
        axi.set(xlabel='Час, с', ylabel='КРП, мкА', title='Сигнали курсового радіоприймача')
        axi.grid(alpha=.3)
        series=[]; max_z=1.; max_t=0.
        for j, name in enumerate(names):
            dense=results[name][2]
            # Для плавного руху вибираємо точки безпосередньо з повного ряду 0.01 с.
            x=dense[:,5]/1000; z=dense[:,6]/1000
            t=dense[:,0]; current=dense[:,7]
            max_z=max(max_z, float(np.max(np.abs(z))))
            max_t=max(max_t, float(t[-1]))
            color=colors[j]
            ax.plot(x[::35], z[::35], color=color, alpha=.22, lw=1.5)
            axi.plot(t[::35], current[::35], color=color, alpha=.22, lw=1.1)
            line,=ax.plot([], [], color=color, lw=2.4, label=name)
            point,=ax.plot([], [], 'o', color=color, ms=8)
            signal,=axi.plot([], [], color=color, lw=1.8)
            signal_point,=axi.plot([], [], 'o', color=color, ms=5)
            series.append((t,x,z,current,line,point,signal,signal_point))
        ax.set_xlim(19,-4.7)
        ax.set_ylim(-max_z*1.22,max_z*1.22)
        axi.set_xlim(0,max_t)
        axi.set_ylim(-270, max(110, max(float(np.max(v[3])) for v in series)*1.12))
        ax.legend(loc='upper right', fontsize=8)
        time_label=fig.text(.5, .025, '', ha='center', fontsize=10)
        def update(frame):
            current_time=max_t*frame/(frames-1)
            for t,x,z,current,line,point,signal,signal_point in series:
                n=int(np.searchsorted(t,current_time,side='right'))
                n=max(1,min(n,len(t)))
                # Скорочуємо кількість точок при рендерингу без зміни даних.
                stride=max(1,n//650)
                ids=np.r_[np.arange(0,n,stride),n-1]
                line.set_data(x[ids],z[ids]); point.set_data([x[n-1]],[z[n-1]])
                signal.set_data(t[ids],current[ids])
                signal_point.set_data([t[n-1]],[current[n-1]])
            time_label.set_text(f'Час моделювання: {current_time:.1f} с')
            return []
        animation=FuncAnimation(fig,update,frames=frames,interval=1000/fps,
                                repeat=True,cache_frame_data=False)
        destination=OUT/f'animation_2D_{group}.gif'
        animation.save(destination,writer=PillowWriter(fps=fps),dpi=95)
        plt.close(fig)
        print('GIF збережено:',destination)

def main():
    results={c.name:run(c) for c in CASES};output(results)
    # Окремі результати ГРМ–ГРП: не втручаємося у базовий розрахунок КРМ.
    for name, (_a, _status, dense) in results.items():
        gr = grp_signal(dense)
        np.savetxt(OUT/f'{name}_GRP_glideslope.csv', gr, delimiter=',',
                   header='t_s,H_m,H_gl_m,epsilon_gl_deg,I_gl_raw_uA,I_GRP_uA,epsilon_gl_filtered_deg',
                   comments='', fmt='%.6f')
    print('Результати дослідів збережено у:',OUT.resolve())
    print('Створено 3 графіки траєкторій, 8 наборів із 6 часових графіків,')
    print('8 таблиць CSV, зведену таблицю та файл коефіцієнтів.')
    import sys
    if '--export-gif' in sys.argv:
        export_group_gifs(results)
    if '--no-gui' not in sys.argv:
        compare_grouped(results, "2d")
        compare_grouped(results, "3d")
        if '--no-graphs' not in sys.argv:
            viewer, radios = graph_viewer(results)
        if '--separate' in sys.argv:
            animation=animate(results)
            view_3d(results)
        plt.show()
if __name__=='__main__':main()
