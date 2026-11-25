from pathlib import Path
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import rcParams
rcParams['font.family']='DejaVu Sans'
rcParams['axes.unicode_minus']=False
DT, V, H0, D0, TAU = 0.01, 78., 500., 15000., 0.2
CASES = [('3.1 — 200 мкА/град',3.,200.),('3.1 — 560 мкА/град',3.,560.),('3.1 — 925 мкА/град',3.,925.),('3.2 — 2°',2.,560.),('3.2 — 3°',3.,560.),('3.2 — 4°',4.,560.)]

def aerodynamic_coefficients():
    # У методичці G0=73000 — польотна вага; у технічній системі m=G0/g.
    S, b, m, iz, rho = 201.45, 5.285, 73000. / 9.81, 660000., .119
    q = rho*V*V/2
    return {
        'c1':13*S*b*b*rho*V/(2*iz),
        'c2':1.51*S*b*q/iz,
        'c3':.96*S*b*q/iz,
        'c4':(5.78+.13)*S*rho*V/(2*m),
        'c5':3.8*S*b*b*rho*V/(2*iz),
        'c6':V/57.3,
        'c9':.2865*S*rho*V/(2*m),
        'c12':.458*S*b*q/iz,
        'c13':1.222*S*rho*V/(2*m),
        'c16':V/(57.3*9.81),
        'c21':2.786*S*b*q/iz,
        'c22':.715*S*rho*V/(2*m),
    }

def clamp(value, limit):
    return float(np.clip(value, -limit, limit))


def lag(x, target, tau):
    return x + DT * (target-x) / tau


def simulate(name, angle, sensitivity):
    """Модель рівнянь та структури САУ зі стор. 3–6 методички.

    F1–F4, F6 — симетричні обмежувачі; 1/(Tp+1) — фільтр,
    kp/(Tp+1) — фільтрована похідна; k/p — інтегратор.
    F_x(Δ) — релейна швидкість тримера ±0.6 град/с за |Δ|>2 град.
    """
    c=aerodynamic_coefficients()
    slope=np.tan(np.deg2rad(angle)); entry=H0/slope-300
    trigger=entry+3500
    d,h=D0,H0
    pitch=q=theta=flap=stab=elevator=receiver=0.0
    trim=0.0; captured=False; rows=[]
    capture_t=None; capture_h=None
    max_after_200=0.0; max_after_capture=0.0; height_at_first_55=None;
    first_below_200=None; over_55_count=0; sample_after_200=0
    receiver=clamp(sensitivity*(np.rad2deg(np.arctan2(h,d+300))-angle),250.)
    # Стан фільтрів та інтеграторів законів керування.
    ih=0.; hfilter=0.; flap_filter=0.
    efilter=0.; pitch_filter11=0.; pitch_filter4=0.; pitch_filter8=0.
    integrator_glide=0.; last_eps=0.; last_pitch=0.; last_h=H0
    glide_lag4=0.; glide_lag8=0.; h_derivative_lag=0.; pitch_lag=0.
    max_steps=int(1.6*D0/V/DT)+200
    for i in range(max_steps):
        t=i*DT
        hg=(d+300)*slope
        eps=np.rad2deg(np.arctan2(h,d+300))-angle
        raw_current=clamp(sensitivity*eps,250.)
        receiver_old=receiver
        receiver=lag(receiver,raw_current,TAU)
        eps_meas=receiver/560.0
        if d<=trigger:
            flap=min(17.,flap+2.*DT)
        if not captured and eps >= 0.0 and d <= entry + 500.0:
            captured=True
            capture_t=t; capture_h=h
            # Перемикання режиму в першій точці перетину глісади.
        stab=-0.14706*flap
        # Перехідні сигнали випуску закрилків: 5 град, T=2 c.
        flap_filter=lag(flap_filter,5.0 if d<=trigger else 0.,2.)
        pitch_rate=q
        if not captured:
            dh=h-H0
            ih += DT*dh
            hfilter=lag(hfilter,dh,1.)
            dh_deriv=(dh-hfilter)/1.
            # Стор. 4–5: Δ*=F1(kI/p ΔH)+kH ΔH+kḢ p/(TḢ p+1) ΔH
            delta_star=clamp(.002*ih,10.)+.2*dh+.4*dh_deriv
            # Δ=F2(Δ*)+kθ pitch+ 1/(Top1 p+1) θop1
            delta=clamp(delta_star,10.)+pitch+flap_filter
            sigma=clamp(q+clamp(delta,8.),10.)
        else:
            if h>250.: k7,k2,t4,f6=15.,210.,2.3,7.
            elif h>=100.: k7,k2,t4,f6=6.5,90.,2.3,3.5
            else: k7,k2,t4,f6=6.5,90.,1.,3.5
            # ε is the filtered receiver current divided by nominal 560 μA/degree.
            efilter=lag(efilter,eps_meas,.7)
            e_deriv=(eps_meas-efilter)/.7
            # θop2=2.5 degrees from capture; in the source it is an additive signal.
            theta_sum=pitch+2.5
            pitch_filter11=lag(pitch_filter11,theta_sum,1.7)
            pitch_d11=(theta_sum-pitch_filter11)/1.7
            # θzad = -F1[k7 ε + k2 p/(T2p+1) ε + k11 p/(T11p+1)(θ+θop2)]
            #         * k4/(T4p+1) + k8 p/(T8p+1)(θ+θop2)
            inner=k7*eps_meas+k2*e_deriv+6.*pitch_d11
            glide_lag4=lag(glide_lag4,inner,t4)
            pitch_filter8=lag(pitch_filter8,theta_sum,15.)
            pitch_d8=(theta_sum-pitch_filter8)/15.
            pitch_set=-clamp(glide_lag4+13.*pitch_d8,7.5)
            # Δ=-F6(kθzad θzad), kθzad=4
            delta=-clamp(4.*pitch_set,f6)
            # σ=F4[(kωz+k*ωz)ωz+Δ], k*ωz=3 after capture.
            sigma=clamp((1.+3.)*q+delta,10.)
        # Релейний інтегратор тримера з рисунка на стор. 4.
        trim_rate=0.6 if delta>2 else (-0.6 if delta< -2 else 0.)
        trim+=DT*trim_rate
        elevator=sigma+trim
        alpha=pitch-theta
        theta_dot=c['c4']*alpha+c['c13']*flap+c['c22']*stab+c['c9']*elevator
        alpha_dot=q-theta_dot
        q_dot=(-c['c1']*q-c['c2']*alpha-c['c5']*alpha_dot
               -c['c12']*flap-c['c21']*stab-c['c3']*elevator)
        if i%500==0 or d<=0:
            rows.append({'t':round(t,2),'D':d,'H':h,'H_glide':hg,'error_h':h-hg,
                'current':receiver,'pitch':pitch,'flap':flap,
                'vertical_speed':c['c6']*theta,'captured':int(captured),
                'elevator_deg':elevator,'stabilizer_deg':stab,
                'theta_deg':theta,'alpha_deg':alpha,'q_deg_s':q,
                'raw_current':raw_current,'trim_deg':trim})
        if captured:
            max_after_capture=max(max_after_capture,abs(receiver))
        if h<=200.0:
            if first_below_200 is None:first_below_200=t
            sample_after_200+=1
            max_after_200=max(max_after_200,abs(receiver))
            if abs(receiver)>55.0:over_55_count+=1
        if d<=0:break
        pitch_old=pitch;theta_old=theta
        pitch+=DT*q
        q+=DT*q_dot
        theta+=DT*theta_dot
        h+=DT*c['c6']*theta_old
        d=max(0.,d-DT*V*np.cos(np.deg2rad(theta_old)))
        if not np.all(np.isfinite([pitch,q,theta,h,d])) or max(abs(pitch),abs(theta),abs(h))>1e5:
            raise RuntimeError(f'{name}: нестійкість моделі на t={t:.2f} с')
    if rows and rows[-1]['D'] > 1.0:
        raise RuntimeError(f'{name}: не досягнуто торця ЗПС')
    # Контроль нормативу виконується за кожним кроком Ейлера, а не за 5-секундним журналом.
    simulate.last_metrics={
        'capture_time_s':capture_t,'capture_height_m':capture_h,
        'max_current_after_capture_uA':max_after_capture,
        'max_current_below_200m_uA':max_after_200,
        'fraction_within_55_below_200m':(1-over_55_count/sample_after_200) if sample_after_200 else None,
        'below_200_time_s':first_below_200,
        'within_55_all_below_200m':bool(sample_after_200 and over_55_count==0),
        'capture_before_200m':bool(capture_t is not None and capture_h>=200),
    }
    return rows,entry



def simulate_full_stabilized(name, angle, sensitivity):
    """Повні рівняння літака зі стор. 3 + експериментальний регулятор.

    ВАЖЛИВО: регулятор не тотожний схемі САУ зі стор. 5 методички.
    Це чисельний експеримент зі стабілізації повної поздовжньої динаміки.
    """
    c = aerodynamic_coefficients()
    slope = np.tan(np.deg2rad(angle))
    entry = H0 / slope - 300.0
    trigger = entry + 3500.0
    d, h = D0, H0
    pitch = q = theta = flap = 0.0
    receiver = clamp(sensitivity * (np.rad2deg(np.arctan2(h, d + 300)) - angle), 250)
    captured = False
    rows = []
    for i in range(26000):
        t = i * DT
        hg = (d + 300) * slope
        eps = np.rad2deg(np.arctan2(h, d + 300)) - angle
        raw = clamp(sensitivity * eps, 250)
        receiver = lag(receiver, raw, TAU)
        if d <= trigger:
            flap = min(17.0, flap + 2.0 * DT)
        stab = -0.14706 * flap
        if not captured and h >= hg and d <= entry + 500:
            captured = True
        # Зовнішній контур: цільовий кут траєкторії.
        if captured:
            theta_target = clamp(-angle - 0.5 * (h - hg), 8.0)
        else:
            theta_target = clamp(0.07 * (H0 - h), 5.0)
        # Балансувальне значення alpha, яке враховує закрилки і стабілізатор.
        denom = c['c4'] - c['c9'] * c['c2'] / c['c3']
        alpha_ff = (-c['c13'] * flap - c['c22'] * stab +
                    c['c9'] * (c['c12'] * flap + c['c21'] * stab) / c['c3']) / denom
        pitch_target = theta_target + alpha_ff
        desired_q_dot = 3.0 * (pitch_target - pitch) - 1.2 * q
        alpha = pitch - theta
        # Керування рулем висоти через повну модель моментів; ±25° —
        # захисне обмеження експерименту, а не норматив методички.
        elevator = clamp(-(desired_q_dot + c['c1'] * q + c['c2'] * alpha +
                           c['c12'] * flap + c['c21'] * stab) / c['c3'], 25.0)
        theta_dot = (c['c4'] * alpha + c['c13'] * flap +
                     c['c22'] * stab + c['c9'] * elevator)
        alpha_dot = q - theta_dot
        q_dot = (-c['c1'] * q - c['c2'] * alpha - c['c5'] * alpha_dot -
                 c['c12'] * flap - c['c21'] * stab - c['c3'] * elevator)
        if i % 500 == 0 or d <= 0:
            rows.append(dict(t=round(t, 2), D=d, H=h, H_glide=hg,
                error_h=h-hg, current=receiver, pitch=pitch, flap=flap,
                vertical_speed=c['c6'] * theta, captured=int(captured),
                elevator_deg=elevator, stabilizer_deg=stab,
                theta_deg=theta, alpha_deg=alpha, q_deg_s=q,
                raw_current=raw, trim_deg=0.0))
        if d <= 0:
            break
        pitch += DT * q
        q += DT * q_dot
        theta_old = theta
        theta += DT * theta_dot
        h += DT * c['c6'] * theta_old
        d = max(0.0, d - DT * V * np.cos(np.deg2rad(theta_old)))
        if not np.all(np.isfinite([pitch, q, theta, h, d])):
            raise RuntimeError('Чисельна нестійкість: ' + name)
    return rows, entry


class Lab44App:
    def __init__(self, root):
        self.root=root
        root.title('Лабораторна 4.4 | ГРМ — поздовжня модель та аналіз нормативів')
        root.geometry('1400x880')
        root.minsize(1080,700)
        self.rows=[]; self.frame=0; self.playing=False; self.job=None
        self.speed=tk.DoubleVar(value=1.0)
        self.orbit=tk.BooleanVar(value=False)
        self.camera_az=-65.0
        self.camera_elev=23.0
        self.case=tk.StringVar(value=CASES[1][0])
        self.mode=tk.StringVar(value='Графіки параметрів')
        self.model=tk.StringVar(value='Закон САУ з методички (діагностика)')
        self.status=tk.StringVar(value='Оберіть дослід і натисніть «Моделювати».')
        bar=ttk.Frame(root,padding=8);bar.pack(fill='x')
        ttk.Label(bar,text='Дослід:').pack(side='left',padx=4)
        ttk.Combobox(bar,textvariable=self.case,values=[c[0] for c in CASES],state='readonly',width=25).pack(side='left',padx=4)
        ttk.Button(bar,text='Моделювати',command=self.run).pack(side='left',padx=5)
        ttk.Combobox(bar,textvariable=self.model,values=['Повна динаміка + експериментальний регулятор','Закон САУ з методички (діагностика)'],state='readonly',width=28).pack(side='left',padx=4)
        ttk.Label(bar,text='Вигляд:').pack(side='left',padx=(20,4))
        selector=ttk.Combobox(bar,textvariable=self.mode,values=['Графіки параметрів','Траєкторія 2D','Траєкторія 3D'],state='readonly',width=22)
        selector.pack(side='left',padx=4);selector.bind('<<ComboboxSelected>>',lambda e:self.draw())
        ttk.Button(bar,text='Експорт CSV',command=self.export).pack(side='left',padx=8)
        ttk.Button(bar,text='Зберегти PNG',command=self.save_png).pack(side='left',padx=4)
        ttk.Button(bar,text='Коефіцієнти',command=self.coeffs).pack(side='left',padx=4)
        ttk.Button(bar,text='Нормативи',command=self.norms).pack(side='left',padx=4)
        self.fig=Figure(figsize=(12,7),dpi=100,facecolor='white')
        self.canvas=FigureCanvasTkAgg(self.fig,master=root)
        self.canvas.get_tk_widget().pack(fill='both',expand=True,padx=8,pady=4)
        self.toolbar=NavigationToolbar2Tk(self.canvas,root,pack_toolbar=False)
        self.toolbar.update();self.toolbar.pack(fill='x',padx=8)
        controls=ttk.Frame(root,padding=(12,8));controls.pack(fill='x')
        self.play_btn=ttk.Button(controls,text='▶ Відтворити',command=self.toggle)
        self.play_btn.pack(side='left',padx=4)
        ttk.Button(controls,text='◀ На початок',command=self.rewind).pack(side='left',padx=4)
        ttk.Label(controls,text='Швидкість:').pack(side='left',padx=(18,4))
        ttk.Combobox(controls,textvariable=self.speed,values=[0.25,0.5,1.0,2.0,4.0,8.0],state='readonly',width=6).pack(side='left')
        ttk.Label(controls,text='3D: обертайте мишею').pack(side='left',padx=10)
        ttk.Button(controls,text='↻ Камера',command=self.reset_camera).pack(side='left',padx=3)
        self.seek=tk.DoubleVar(value=0)
        self.scale=ttk.Scale(controls,from_=0,to=100,variable=self.seek,command=self.scrub)
        self.scale.pack(side='left',fill='x',expand=True,padx=18)
        self.time_label=ttk.Label(controls,text='t = 0,0 с',width=16)
        self.time_label.pack(side='right')
        ttk.Label(root,textvariable=self.status,anchor='w',padding=8).pack(fill='x')
        self.log=tk.Text(root,height=5,wrap='none',font=('Consolas',9))
        self.log.pack(fill='x',padx=10,pady=(0,8))
        self.log.insert('end','Журнал: дані виводяться з інтервалом 5 с.\n')
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        self.run()
    def run(self):
        self.playing=False;self.play_btn.configure(text='▶ Відтворити')
        name,angle,sensitivity=next(x for x in CASES if x[0]==self.case.get())
        try:self.rows,self.entry=(simulate_full_stabilized if self.model.get().startswith('Повна динаміка') else simulate)(name,angle,sensitivity)
        except Exception as exc:
            messagebox.showerror('Помилка чисельного моделювання',str(exc));return
        self.metrics = simulate.last_metrics if not self.model.get().startswith('Повна динаміка') else None
        self.frame=0;self.scale.configure(to=max(1,len(self.rows)-1));self.seek.set(0)
        r=self.rows;final=r[-1];captured=next((x for x in r if x['captured']),None)
        self.log.delete('1.0','end')
        self.log.insert('end',' t, с | D, м | H, м | ΔHгл, м | Iгрп, мкА | δв, ° | δз, ° | φст, ° | тангаж, °\n')
        for x in r:
            self.log.insert('end',f"{x['t']:6.1f} | {x['D']:7.1f} | {x['H']:7.2f} | {x['error_h']:+8.2f} | {x['current']:+10.2f} | {x['elevator_deg']:+6.2f} | {x['flap']:5.1f} | {x['stabilizer_deg']:+6.2f} | {x['pitch']:+7.2f}\n")
        self.status.set('')
        self.fig.clear();self._scene3d=None
        self.draw()
    def arrays(self):
        return {k:np.array([r[k] for r in self.rows],float) for k in self.rows[0]}
    def draw(self):
        if not self.rows:return
        # Зберігаємо ракурс, який користувач встановив мишею, перед очищенням осей.
        old_3d = next((axis for axis in self.fig.axes if getattr(axis, 'name', '') == '3d'), None)
        if old_3d is not None:
            self.camera_az = float(old_3d.azim)
            self.camera_elev = float(old_3d.elev)
        a=self.arrays();mode=self.mode.get();j=min(self.frame,len(self.rows)-1)
        if mode != 'Траєкторія 3D' or not any(getattr(axis,'name','')=='3d' for axis in self.fig.axes):
            self.fig.clear();self._scene3d=None
        t=a['t'];d=a['D']/1000;h=a['H'];hg=a['H_glide'];I=a['current']
        if mode=='Графіки параметрів':
            axs=self.fig.subplots(2,2)
            ax=axs[0,0];ax.plot(d,h,label='Літак');ax.plot(d,hg,'--',label='Глісада');ax.scatter([-.3],[0],marker='^',s=95,c='darkgreen',zorder=9,label='ГРМ (300 м за торцем ЗПС)');ax.annotate('ГРМ',(-.3,0),xytext=(-12,13),textcoords='offset points',color='darkgreen');ax.invert_xaxis();ax.set(xlabel='Відстань до ЗПС, км',ylabel='Висота, м');ax.legend()
            ax=axs[0,1];ax.plot(t,a['error_h']);ax.axhline(0,ls=':',c='gray');ax.set(xlabel='Час, с',ylabel='Відхилення від глісади, м')
            ax=axs[1,0];ax.plot(t,I,label='Струм ГРП');ax.axhline(55,ls=':',c='gray');ax.axhline(-55,ls=':',c='gray');ax.axhline(250,ls='--',c='red',alpha=.6);ax.axhline(-250,ls='--',c='red',alpha=.6);ax.set(xlabel='Час, с',ylabel='Струм, мкА');ax.legend()
            ax=axs[1,1]
            for k,label in [('pitch','Тангаж'),('theta_deg','Кут траєкторії'),('elevator_deg','Руль висоти'),('flap','Закрилки')]:ax.plot(t,a[k],label=label)
            ax.set(xlabel='Час, с',ylabel='Градуси');ax.legend(fontsize=8)
            for ax in axs.flat:ax.grid(alpha=.25)
        elif mode=='Траєкторія 2D':
            ax=self.fig.add_subplot(111)
            ax.plot(d,hg,'--',c='tab:orange',label='Глісада')
            ax.plot(d,h,c='tab:blue',alpha=.18,label='Повний шлях')
            ax.plot(d[:j+1],h[:j+1],c='tab:blue',lw=2.5,label='Пройдений шлях')
            # Бічний силует літака: носова частина спрямована до ЗПС.
            # Масштаб умовний: силует має залишатися видимим на осі км/м.
            plane_length = .72  # км (візуальний масштаб)
            plane_height = 29.0  # м (візуальний масштаб)
            pitch = np.deg2rad(float(a['pitch'][j]))
            # Локальні координати: x уперед, y вгору; невеликий нахил за тангажем.
            def plane_points(coords):
                xy = np.asarray(coords, dtype=float)
                x0 = xy[:, 0] * np.cos(pitch) - xy[:, 1] * np.sin(pitch)
                y0 = xy[:, 0] * np.sin(pitch) + xy[:, 1] * np.cos(pitch)
                return np.column_stack((d[j] - x0 * plane_length,
                                        h[j] + y0 * plane_height))
            def add_part(coords, face, edge='#17324b', lw=1.0, z=10):
                ax.add_patch(Polygon(plane_points(coords), closed=True,
                                     facecolor=face, edgecolor=edge, lw=lw,
                                     zorder=z, joinstyle='round'))
            # Фюзеляж, хвостове оперення, основне крило, кабіна й ілюмінатори.
            add_part([(-.94,-.09),(-.72,-.13),(-.35,-.16),(.48,-.14),(.78,-.07),
                      (1.00,.00),(.80,.10),(.38,.14),(-.44,.14),(-.78,.10)], '#2f78b6')
            add_part([(-.84,.06),(-.86,.52),(-.67,.54),(-.46,.12)], '#4f9ad1', z=11)
            add_part([(-.20,-.08),(-.42,-.39),(-.08,-.39),(.40,-.07)], '#185b92', z=12)
            add_part([(.53,.08),(.75,.08),(.87,.025),(.57,.025)], '#b8e3f5',
                     edge='#0d4a75', lw=.8, z=13)
            for px in (-.35,-.18,-.01,.16,.33):
                add_part([(px,.055),(px+.07,.055),(px+.07,.09),(px,.09)],
                         '#d8f2ff', edge='#d8f2ff', lw=.2, z=13)
            # ЗПС як реальна смуга: асфальт, бокові лінії, осьова розмітка,
            # поріг та смуги приземлення. Довжина 1.8 км показана умовно.
            runway_start, runway_end = -1.8, 0.0
            runway_top, runway_bottom = 0.0, -13.0
            ax.add_patch(Polygon([(runway_start,runway_bottom),(runway_end,runway_bottom),
                                  (runway_end,runway_top),(runway_start,runway_top)],
                                 closed=True,facecolor='#454b52',edgecolor='#1e252a',
                                 linewidth=1.5,zorder=2,label='ЗПС'))
            ax.plot([runway_start,runway_end],[runway_top-1.5]*2,
                    c='white',lw=1.4,zorder=4)
            ax.plot([runway_start,runway_end],[runway_bottom+1.5]*2,
                    c='white',lw=1.4,zorder=4)
            for mid in np.arange(-1.65,-.08,.24):
                ax.plot([mid,mid+.12],[-6.5,-6.5],c='white',lw=2.5,
                        solid_capstyle='butt',zorder=5)
            for yy in (-3.0,-4.6,-8.4,-10.0):
                ax.plot([-.055,-.055],[yy-.6,yy+.6],c='white',lw=2,zorder=5)
            ax.plot([0,0],[runway_bottom,runway_top],c='white',lw=2.5,zorder=6)
            ax.annotate('ЗПС',(-1.35,-6.5),color='white',fontsize=9,
                        fontweight='bold',ha='center',va='center',zorder=7)
            # ГРМ розташовано на 300 м за торцем ЗПС (D = -300 м).
            ax.scatter([-.3],[0],marker='^',s=130,c='darkgreen',zorder=15,label='ГРМ')
            ax.annotate('ГРМ',(-.3,0),xytext=(-17,16),textcoords='offset points',
                        color='darkgreen',fontsize=11,fontweight='bold',zorder=16)
            ax.invert_xaxis();ax.set(xlabel='Відстань до торця ЗПС, км',ylabel='Висота, м')
            ax.set_xlim(max(d)+.5,-2.05)
            ax.set_ylim(min(-20,h.min(),hg.min())-15,max(h.max(),hg.max())+60)
            ax.grid(alpha=.3);ax.legend()
        else:
            # ВАЖЛИВО: не видаляємо й не очищуємо Axes3D під час анімації.
            # Інакше Matplotlib втрачає активне перетягування мишею.
            x=-d; y=np.zeros_like(x)
            scene=getattr(self,'_scene3d',None)
            if scene is None:
                ax=self.fig.add_subplot(111,projection='3d')
                ax.plot(x,y,hg,'--',c='tab:orange',label='Глісада')
                ax.plot(x,y,h,c='tab:blue',alpha=.18)
                trail,=ax.plot([],[],[],c='tab:blue',lw=2.5,label='Пройдений шлях')
                meshes=[]
                for color in ('#2e78b5','#75b7e1','#b7d7ec','#155b90'):
                    mesh=Poly3DCollection([[(0,0,0)]*3],facecolors=color,
                                          edgecolors='#173b55',linewidths=.8,zsort='average')
                    ax.add_collection3d(mesh)
                    meshes.append(mesh)
                # Горизонтальна поверхня ЗПС з білою розміткою.
                runway=Poly3DCollection([[(0,-.13,0),(1.8,-.13,0),
                                           (1.8,.13,0),(0,.13,0)]],
                                          facecolors='#454b52',edgecolors='#24292f',
                                          linewidths=1.2,alpha=1.0)
                ax.add_collection3d(runway)
                for yy in (-.115,.115):
                    ax.plot([0,1.8],[yy,yy],[1,1],c='white',lw=1.6)
                for mid in np.arange(.13,1.7,.25):
                    ax.plot([mid,mid+.12],[0,0],[2,2],c='white',lw=2.3)
                for yy in (-.075,-.04,.04,.075):
                    ax.plot([.035,.10],[yy,yy],[2,2],c='white',lw=2.3)
                ax.text(1.0,0,5,'ЗПС',color='white',fontsize=9,ha='center')
                # Нерухомий наземний маяк ГРМ за 300 м від торця ЗПС.
                ax.scatter([.3],[.23],[0],c='darkgreen',marker='^',s=85,depthshade=False,label='ГРМ')
                ax.plot([.3,.3],[.23,.23],[0,16],c='darkgreen',lw=2)
                ax.text(.3,.23,24,'ГРМ',color='darkgreen',fontsize=10,fontweight='bold')
                ax.set(xlabel='Поздовжня координата, км',ylabel='Бічна координата, км',zlabel='Висота, м')
                ax.set_xlim(-15.5,2.1);ax.set_ylim(-.7,.7)
                ax.set_zlim(min(0,h.min())-20,max(h.max(),hg.max())+60)
                ax.view_init(elev=self.camera_elev,azim=self.camera_az)
                ax.set_box_aspect((13,3.5,5),zoom=.92)
                ax.legend(loc='upper left',fontsize=8)
                scene=(ax,trail,meshes)
                self._scene3d=scene
            ax,trail,meshes=scene
            trail.set_data_3d(x[:j+1],y[:j+1],h[:j+1])
            cx,cy,cz=float(x[j]),0.,float(h[j])
            pitch=np.deg2rad(float(a['pitch'][j]));scale=.30
            def xyz(longitudinal,lateral,vertical=0.):
                xx=longitudinal*np.cos(pitch)-vertical*np.sin(pitch)
                zz=longitudinal*np.sin(pitch)+vertical*np.cos(pitch)
                return (cx+xx*scale,cy+lateral*scale,cz+zz*scale*100.)
            fuselage=[xyz(.95,0,0),xyz(.40,.09,.03),xyz(-.70,.09,0),xyz(-.95,0,0),xyz(-.70,-.09,0),xyz(.40,-.09,.03)]
            wings=[xyz(.10,-.9,0),xyz(-.30,-.9,0),xyz(-.43,0,0),xyz(-.30,.9,0),xyz(.10,.9,0),xyz(.28,0,0)]
            tail=[xyz(-.65,-.42,0),xyz(-.85,-.42,0),xyz(-.95,0,0),xyz(-.85,.42,0),xyz(-.65,.42,0)]
            fin=[xyz(-.60,0,0),xyz(-.82,0,.55),xyz(-.95,0,.52),xyz(-.95,0,0)]
            for mesh,verts in zip(meshes,(fuselage,wings,tail,fin)):
                mesh.set_verts([verts])
            # Не викликаємо view_init: ракурс належить користувачу.
        self.fig.suptitle(f'{self.case.get()} | t = {t[j]:.1f} с | H = {h[j]:.1f} м | I = {I[j]:+.1f} мкА',fontsize=12)
        self.fig.subplots_adjust(left=.08,right=.96,bottom=.11,top=.90,wspace=.3,hspace=.35)
        self.canvas.draw_idle();self.time_label.configure(text=f't = {t[j]:.1f} с')
    def reset_camera(self):
        self.camera_az=-65.0
        self.camera_elev=23.0
        # Не даємо draw() зчитати старий ракурс перед скиданням.
        self.fig.clear()
        self._scene3d=None
        self.draw()

    def scrub(self,value):
        if not self.rows:return
        j=max(0,min(len(self.rows)-1,int(round(float(value)))))
        if j!=self.frame:self.frame=j;self.draw()
    def toggle(self):
        if not self.rows:return
        self.playing=not self.playing
        self.play_btn.configure(text='⏸ Пауза' if self.playing else '▶ Відтворити')
        if self.playing:self.tick()
    def tick(self):
        if not self.playing:return
        self.frame+=max(1,round(self.speed.get()))
        if self.frame>=len(self.rows):self.frame=0
        self.seek.set(self.frame)
        self.draw()
        self.job=self.root.after(120,self.tick)
    def rewind(self):
        self.playing=False;self.play_btn.configure(text='▶ Відтворити');self.frame=0;self.seek.set(0);self.draw()
    def norms(self):
        if not self.rows:return
        if self.metrics is None:
            messagebox.showinfo('Нормативи','Для перевірки за кожним кроком оберіть «Закон САУ з методички (діагностика)» і натисніть «Моделювати».')
            return
        m=self.metrics
        txt=(f"Захоплення: t={m['capture_time_s']:.2f} с, H={m['capture_height_m']:.2f} м\n"
             f"Захоплення до H=200 м: {'ТАК' if m['capture_before_200m'] else 'НІ'}\n"
             f"Макс. |I| після захоплення: {m['max_current_after_capture_uA']:.2f} мкА\n"
             f"Макс. |I| при H≤200 м: {m['max_current_below_200m_uA']:.2f} мкА\n"
             f"Частка кроків у ±55 мкА при H≤200 м: {m['fraction_within_55_below_200m']*100:.2f}%\n"
             f"Вимога ±55 мкА при H≤200 м: {'ТАК' if m['within_55_all_below_200m'] else 'НІ'}\n"
             'Примітка: оцінка перерегулювання ±125 мкА потребує окремого визначення фази переходу.')
        messagebox.showinfo('Чисельна перевірка нормативів',txt)
    def coeffs(self):
        c=aerodynamic_coefficients()
        messagebox.showinfo('Коефіцієнти поздовжньої моделі','\n'.join(f'{k} = {v:.8f}' for k,v in c.items()))
    def export(self):
        if not self.rows:return
        path=filedialog.asksaveasfilename(defaultextension='.csv',filetypes=[('CSV','*.csv')],initialfile='lab44_results.csv')
        if path:
            with open(path,'w',encoding='utf-8-sig',newline='') as f:
                w=csv.DictWriter(f,fieldnames=self.rows[0].keys());w.writeheader();w.writerows(self.rows)
            self.status.set('Збережено: '+path)
    def save_png(self):
        path=filedialog.asksaveasfilename(defaultextension='.png',filetypes=[('PNG','*.png')],initialfile='lab44_graph.png')
        if path:self.fig.savefig(path,dpi=180);self.status.set('Збережено: '+path)
    def close(self):
        self.playing=False
        if self.job is not None:
            try:self.root.after_cancel(self.job)
            except Exception:pass
        self.root.destroy()

if __name__=='__main__':
    root=tk.Tk()
    try:
        ttk.Style(root).theme_use('clam')
    except tk.TclError:pass
    Lab44App(root)
    root.mainloop()
