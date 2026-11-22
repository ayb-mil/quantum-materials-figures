"""
make_figures.py
===============
Reproducible figure suite for the M.Sc. thesis

    "Nanomaterials for the Hardware of Quantum Devices — From Material
     Limitations of Existing Platforms to Defect Engineering in
     Hexagonal Boron Nitride"
    Ayoub Elmqarta, ENS Meknes / Moulay Ismail University (2026)

Running this script writes 21 PNG figures to ./figures/

    pip install -r requirements.txt
    python make_figures.py

Figure index
------------
  S1   Transmon anharmonicity                   §1.2.1 Superconducting qubits
  S2   Surface-code below-threshold scaling     §1.2.1 / §1.3.3
  S3   Zeeman splitting of a spin-1/2           §1.2.2 Semiconductor spin qubits
  S4   Aluminium (FCC)                          §2.1.1 Superconducting materials
  S5   Niobium (BCC)                            §2.1.1
  S6   NbN (rocksalt B1)                        §2.1.1
  S7   TiN (rocksalt B1)                        §2.1.1
  S8   Silicon (diamond cubic)                  §2.1.2 Semiconductor materials
  S9   Germanium (diamond cubic)                §2.1.2
  S10  GaAs (zincblende)                        §2.1.2
  S11  InAs (zincblende)                        §2.1.2
  S12  Bi2Te3 + Dirac surface state             §2.1.3 Topological / hybrid
  S13  InAs/Al hybrid nanowire                  §2.1.3
  S14  Diamond NV- centre + level scheme        §2.1.4 Photonic materials
  S15  Si3N4 / SiO2 photonic platforms          §2.1.4
  S20  Stress-strain: hBN vs steel              §2.5   hBN mechanical properties
  S21  V_B- defect in the hBN honeycomb         §2.5.2 Defect-based quantum emitters
  M1   Coherence time vs material strategy      §1.3.1 Coherence
  M2   1/f two-level-system noise               §1.3.2 Noise
  M3   Gate-error progression vs threshold      §1.3.3 Error rates
  M4   Scalability: JJ disorder + qubit count   §1.3.4 Scalability

Numerical values are taken from the literature cited in the thesis.
M1-M4 are schematic syntheses: values are order-of-magnitude, see README.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, Patch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from itertools import product

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   11,
    "axes.linewidth": 1.2,
})

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

C_SC, C_SPIN, C_DEFECT, C_TI = "#c0392b", "#2874a6", "#27ae60", "#8e44ad"
C_TARGET, C_GRID = "#1B3A5C", "#cccccc"


# ====================================================================
# Helpers for 3-D crystal panels (used by S4-S11)
# ====================================================================
def draw_crystal(ax, positions, colors, sizes, bonds=None, a=1.0, title=""):
    xs, ys, zs = zip(*positions)
    ax.scatter(xs, ys, zs, c=colors, s=sizes, edgecolors="black",
               linewidths=0.6, depthshade=True)
    if bonds is not None:
        segs = [[positions[i], positions[j]] for (i, j) in bonds]
        ax.add_collection3d(Line3DCollection(segs, colors="#666",
                                             linewidths=1.0, alpha=0.7))
    for s, e in [
        ([0,0,0],[a,0,0]),([0,0,0],[0,a,0]),([0,0,0],[0,0,a]),
        ([a,a,a],[0,a,a]),([a,a,a],[a,0,a]),([a,a,a],[a,a,0]),
        ([a,0,0],[a,a,0]),([a,0,0],[a,0,a]),
        ([0,a,0],[a,a,0]),([0,a,0],[0,a,a]),
        ([0,0,a],[a,0,a]),([0,0,a],[0,a,a]),
    ]:
        ax.plot(*zip(s, e), color="gray", lw=0.6, alpha=0.5)
    ax.set_xlim(-0.1,a+0.1); ax.set_ylim(-0.1,a+0.1); ax.set_zlim(-0.1,a+0.1)
    ax.set_box_aspect([1,1,1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.xaxis.pane.set_visible(False); ax.yaxis.pane.set_visible(False); ax.zaxis.pane.set_visible(False)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=-8)

def info_panel(ax, title, rows):
    ax.axis("off")
    ax.text(0.02, 0.95, title, fontsize=13, fontweight="bold",
            transform=ax.transAxes, color="#1B3A5C")
    y = 0.85
    for label, value in rows:
        ax.text(0.02, y, label, fontsize=9.5, fontweight="bold",
                transform=ax.transAxes, color="#333")
        ax.text(0.42, y, value, fontsize=9.5, transform=ax.transAxes, color="#111")
        y -= 0.085

def fcc_positions(a=1.0):
    corners = list(product([0,a], repeat=3))
    faces = [(a/2,a/2,0),(a/2,a/2,a),(a/2,0,a/2),(a/2,a,a/2),(0,a/2,a/2),(a,a/2,a/2)]
    return corners + faces

def bcc_positions(a=1.0):
    return list(product([0,a], repeat=3)) + [(a/2,a/2,a/2)]

def rocksalt_positions(a=1.0):
    A = fcc_positions(a)
    edge = [(a/2,0,0),(a/2,a,0),(a/2,0,a),(a/2,a,a),
            (0,a/2,0),(a,a/2,0),(0,a/2,a),(a,a/2,a),
            (0,0,a/2),(a,0,a/2),(0,a,a/2),(a,a,a/2)]
    B = edge + [(a/2,a/2,a/2)]
    return A, B

def diamond_positions(a=1.0):
    A = fcc_positions(a)
    B = [(x+a/4,y+a/4,z+a/4) for (x,y,z) in A
         if x+a/4 <= a and y+a/4 <= a and z+a/4 <= a]
    return A, B

def diamond_bonds(A, B, a=1.0):
    bonds = []
    nA = len(A); expected = a*np.sqrt(3)/4
    for i,p in enumerate(A):
        for j,q in enumerate(B):
            if abs(np.linalg.norm(np.array(p)-np.array(q)) - expected) < 0.05*a:
                bonds.append((i, nA+j))
    return bonds


# ====================================================================
# S1 - Transmon anharmonicity
# ====================================================================
def fig_s1_transmon():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    levels_HO = [0,1,2,3,4]; alpha=-0.25
    levels_TR=[0]; E=0
    for k in range(4):
        E += 1.0 + k*alpha; levels_TR.append(E)
    x1,x2,x3,x4 = 0.0,1.0,2.2,3.2
    for i,E in enumerate(levels_HO):
        ax.hlines(E,x1,x2,color="#2874a6",lw=2); ax.text(x2+0.08,E,f"|{i}⟩",va="center")
    for i,E in enumerate(levels_TR):
        ax.hlines(E,x3,x4,color="#c0392b",lw=2); ax.text(x4+0.08,E,f"|{i}⟩",va="center")
    ax.annotate("",xy=(0.5,1),xytext=(0.5,0),arrowprops=dict(arrowstyle="->",color="black",lw=1.5))
    ax.text(0.55,0.5,r"$\omega_{01}$")
    ax.annotate("",xy=(0.5,2),xytext=(0.5,1),arrowprops=dict(arrowstyle="->",color="black",lw=1.5))
    ax.text(0.55,1.5,r"$\omega_{12}=\omega_{01}$",color="#c0392b")
    ax.annotate("",xy=(2.7,levels_TR[1]),xytext=(2.7,0),arrowprops=dict(arrowstyle="->",color="black",lw=1.5))
    ax.text(2.75,0.5,r"$\omega_{01}$")
    ax.annotate("",xy=(2.7,levels_TR[2]),xytext=(2.7,levels_TR[1]),arrowprops=dict(arrowstyle="->",color="black",lw=1.5))
    ax.text(2.75,levels_TR[1]+0.25,r"$\omega_{12}=\omega_{01}+\alpha$",color="green")
    ax.text(0.5,-0.5,"Harmonic LC\n(no qubit possible)",ha="center",color="#2874a6",fontsize=10)
    ax.text(2.7,-0.5,"Transmon (Josephson)\nα ≈ −200…−300 MHz",ha="center",color="#c0392b",fontsize=10)
    ax.set_ylim(-1,4.5); ax.set_xlim(-0.5,4.5); ax.axis("off")
    ax.set_title("Fig. S1 — Why a superconducting qubit needs anharmonicity")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_S1_transmon.png",dpi=160); plt.close()

# ====================================================================
# S2 - Surface-code below-threshold scaling
# ====================================================================
def fig_s2_threshold():
    fig,ax = plt.subplots(figsize=(7.5,4.5))
    p_phys = np.array([0.3,0.5,0.7,1.0]); d = np.array([3,5,7,9])
    for p in p_phys:
        pL = 0.03*(p/1.0)**((d+1)/2)
        ax.plot(d,pL,marker="o",lw=2,label=f"p_phys = {p}%")
    ax.axhline(1e-6,color="green",ls="--",label="target logical error 10⁻⁶")
    ax.set_yscale("log")
    ax.set_xlabel("Surface-code distance d"); ax.set_ylabel("Logical error rate p_L per cycle")
    ax.set_title("Fig. S2 — Below-threshold exponential suppression of logical errors\n"
                 "(surface-code threshold ≈ 1%)")
    ax.legend(fontsize=9,loc="upper right"); ax.grid(alpha=0.3,which="both")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_S2_threshold.png",dpi=160); plt.close()

# ====================================================================
# S3 - Zeeman splitting
# ====================================================================
def fig_s3_zeeman():
    fig,ax = plt.subplots(figsize=(7,4))
    Bs = np.linspace(0,1,100); g,mu_B,h = 2.0,9.274e-24,6.626e-34
    dE = g*mu_B*Bs/h*1e-9
    ax.plot(Bs,+0.5*dE,color="#c0392b",lw=2,label=r"$m_s=+1/2$")
    ax.plot(Bs,0*Bs,color="gray",lw=1,ls=":",label="B = 0 (degenerate)")
    ax.plot(Bs,-0.5*dE,color="#2874a6",lw=2,label=r"$m_s=-1/2$")
    ax.fill_between(Bs,-0.5*dE,+0.5*dE,color="#f2eef5",alpha=0.4)
    ax.annotate("",xy=(0.75,+0.5*dE[75]),xytext=(0.75,-0.5*dE[75]),
                arrowprops=dict(arrowstyle="<->",color="black",lw=1.4))
    ax.text(0.78,0,r"$\Delta E = g\,\mu_B B$",fontsize=13)
    ax.set_xlabel("Magnetic field B (T)"); ax.set_ylabel(r"Energy $\Delta E/h$ (GHz)")
    ax.set_title("Fig. S3 — Zeeman splitting of a spin-1/2 state")
    ax.legend(loc="upper left"); ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_S3_zeeman.png",dpi=160); plt.close()

# ====================================================================
# S4 - Aluminium (FCC)
# ====================================================================
def fig_s4_al():
    fig = plt.figure(figsize=(9,4.2))
    ax1 = fig.add_subplot(121, projection="3d")
    pos = fcc_positions()
    draw_crystal(ax1,pos,["#b3b3b3"]*len(pos),[200]*len(pos),title="Al — FCC")
    ax1.view_init(elev=20,azim=30)
    ax2 = fig.add_subplot(122)
    info_panel(ax2,"Aluminum (Al)",[
        ("Crystal system:","FCC,  a = 4.05 Å"),
        ("Superconducting T_c:","1.20 K"),
        ("Critical field B_c:","10.5 mT"),
        ("London penetration λ:","~16 nm"),
        ("Coherence length ξ:","~1.6 μm"),
        ("Quantum role:","Transmon electrodes"),
        ("Josephson junction:","Al / AlO_x / Al trilayer"),
        ("Key source:","Kittel (2005); Place 2021"),
    ])
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_S4_al.png",dpi=150,bbox_inches="tight"); plt.close()

# ====================================================================
# S5 - Niobium (BCC)
# ====================================================================
def fig_s5_nb():
    fig = plt.figure(figsize=(9,4.2))
    ax1 = fig.add_subplot(121, projection="3d")
    pos = bcc_positions(); cols = ["#4a6c8a"]*8 + ["#1B3A5C"]
    draw_crystal(ax1,pos,cols,[200]*8+[260],title="Nb — BCC")
    ax1.view_init(elev=20,azim=30)
    ax2 = fig.add_subplot(122)
    info_panel(ax2,"Niobium (Nb)",[
        ("Crystal system:","BCC,  a = 3.30 Å"),
        ("Superconducting T_c:","9.25 K (highest of elements)"),
        ("Thermodynamic B_c:","~ 0.20 T (200 mT)"),
        ("Upper critical B_c2:","~ 0.40 T (type-II)"),
        ("London penetration λ:","~ 40 nm"),
        ("Coherence length ξ:","~ 38 nm"),
        ("Quantum role:","CPW resonators, flux qubits"),
        ("Loss tangent:","tan δ ≈ 10⁻⁶ at 20 mK"),
        ("Key source:","Poole 2000; Welander 2013"),
    ])
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_S5_nb.png",dpi=150,bbox_inches="tight"); plt.close()

# ====================================================================
# S6 - NbN (rocksalt B1)
# ====================================================================
def fig_s6_nbn():
    fig = plt.figure(figsize=(9,4.2))
    ax1 = fig.add_subplot(121,projection="3d")
    A,B = rocksalt_positions(); pos=A+B
    cols = ["#4a6c8a"]*len(A)+["#d4a017"]*len(B)
    sizes = [180]*len(A)+[140]*len(B)
    draw_crystal(ax1,pos,cols,sizes,title="NbN — Rocksalt (B1)")
    ax1.view_init(elev=20,azim=30)
    ax1.scatter([],[],[],c="#4a6c8a",s=60,label="Nb"); ax1.scatter([],[],[],c="#d4a017",s=60,label="N")
    ax1.legend(loc="upper right",fontsize=8,frameon=False)
    ax2 = fig.add_subplot(122)
    info_panel(ax2,"Niobium nitride (NbN)",[
        ("Crystal system:","Rocksalt B1,  a = 4.39 Å"),
        ("Superconducting T_c:","≈ 16 K (thin films)"),
        ("Critical field B_c2:","~15 T"),
        ("Kinetic inductance:","~80–200 pH/□"),
        ("Gap (2Δ/k_BT_c):","~4.0 (strong-coupling)"),
        ("Quantum role:","KIDs, high-Z resonators"),
        ("","SNSPD detectors"),
        ("Key source:","Annunziata 2010; Frasca 2023"),
    ])
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_S6_nbn.png",dpi=150,bbox_inches="tight"); plt.close()

# ====================================================================
# S7 - TiN (rocksalt B1)
# ====================================================================
def fig_s7_tin():
    fig = plt.figure(figsize=(9,4.2))
    ax1 = fig.add_subplot(121,projection="3d")
    A,B = rocksalt_positions(); pos=A+B
    cols = ["#c0c0c0"]*len(A)+["#d4a017"]*len(B)
    sizes = [180]*len(A)+[140]*len(B)
    draw_crystal(ax1,pos,cols,sizes,title="TiN — Rocksalt (B1)")
    ax1.view_init(elev=20,azim=30)
    ax1.scatter([],[],[],c="#c0c0c0",s=60,label="Ti"); ax1.scatter([],[],[],c="#d4a017",s=60,label="N")
    ax1.legend(loc="upper right",fontsize=8,frameon=False)
    ax2 = fig.add_subplot(122)
    info_panel(ax2,"Titanium nitride (TiN)",[
        ("Crystal system:","Rocksalt B1,  a = 4.24 Å"),
        ("Superconducting T_c:","4.5–5.6 K (thin films)"),
        ("Kinetic inductance:","~30–100 pH/□"),
        ("Loss tangent:","tan δ ≈ 10⁻⁷ at mK"),
        ("Dielectric match:","Excellent Si / SiN compatibility"),
        ("Quantum role:","Low-loss resonators, MKIDs,"),
        ("","transmon capacitor pads"),
        ("Key source:","Vissers 2010; Shearrow 2018"),
    ])
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_S7_tin.png",dpi=150,bbox_inches="tight"); plt.close()

# ====================================================================
# S8 - Silicon (diamond cubic)
# ====================================================================
def fig_s8_si():
    fig = plt.figure(figsize=(9,4.2))
    ax1 = fig.add_subplot(121,projection="3d")
    A,B = diamond_positions(); pos = A+B
    draw_crystal(ax1,pos,["#8a8a8a"]*(len(A)+len(B)),[170]*(len(A)+len(B)),
                 bonds=diamond_bonds(A,B),title="Si — Diamond cubic")
    ax1.view_init(elev=20,azim=30)
    ax2 = fig.add_subplot(122)
    info_panel(ax2,"Silicon (Si)",[
        ("Crystal system:","Diamond cubic, a = 5.43 Å"),
        ("Band gap:","1.12 eV (indirect)"),
        ("Nuclear spin (²⁹Si):","I = 1/2, 4.7% nat. abundance"),
        ("²⁸Si spin-free host:","Enriched → QD T₂* ~120 µs"),
        ("Mobility (electron):","1400 cm²/(V·s) at 300 K"),
        ("Quantum role:","Spin qubits in SiMOS, Si/SiGe QDs"),
        ("Gate fidelity:","> 99.9% single-qubit"),
        ("Key source:","Veldhorst 2014; Noiri 2022"),
    ])
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_S8_si.png",dpi=150,bbox_inches="tight"); plt.close()

# ====================================================================
# S9 - Germanium (diamond cubic)
# ====================================================================
def fig_s9_ge():
    fig = plt.figure(figsize=(9,4.2))
    ax1 = fig.add_subplot(121,projection="3d")
    A,B = diamond_positions(); pos=A+B
    draw_crystal(ax1,pos,["#c8823c"]*(len(A)+len(B)),[170]*(len(A)+len(B)),
                 bonds=diamond_bonds(A,B),title="Ge — Diamond cubic")
    ax1.view_init(elev=20,azim=30)
    ax2 = fig.add_subplot(122)
    info_panel(ax2,"Germanium (Ge)",[
        ("Crystal system:","Diamond cubic, a = 5.66 Å"),
        ("Band gap:","0.66 eV (indirect)"),
        ("Nuclear spin (⁷³Ge):","I = 9/2, 7.8% abundance"),
        ("Hole g-factor:","Anisotropic, tunable"),
        ("Hole mobility:","~1900 cm²/(V·s) > Si"),
        ("Spin-orbit:","Strong; all-electrical control"),
        ("Quantum role:","Planar Ge / Ge/SiGe hole qubits"),
        ("Key source:","Jirovec 2021; Hendrickx 2020"),
    ])
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_S9_ge.png",dpi=150,bbox_inches="tight"); plt.close()

# ====================================================================
# S10 - GaAs (zincblende)
# ====================================================================
def fig_s10_gaas():
    fig = plt.figure(figsize=(9,4.2))
    ax1 = fig.add_subplot(121,projection="3d")
    A,B = diamond_positions(); pos=A+B
    cols = ["#c8823c"]*len(A)+["#6aa84f"]*len(B)
    sizes = [170]*len(A)+[160]*len(B)
    draw_crystal(ax1,pos,cols,sizes,bonds=diamond_bonds(A,B),title="GaAs — Zincblende")
    ax1.view_init(elev=20,azim=30)
    ax1.scatter([],[],[],c="#c8823c",s=60,label="Ga"); ax1.scatter([],[],[],c="#6aa84f",s=60,label="As")
    ax1.legend(loc="upper right",fontsize=8,frameon=False)
    ax2 = fig.add_subplot(122)
    info_panel(ax2,"Gallium Arsenide (GaAs)",[
        ("Crystal system:","Zincblende, a = 5.65 Å"),
        ("Band gap:","1.42 eV (direct)"),
        ("Nuclear spin:","100% of ⁶⁹Ga, ⁷¹Ga, ⁷⁵As"),
        ("Mobility (electron):","8500 cm²/(V·s)"),
        ("Historical issue:","Nuclear-spin decoherence"),
        ("Recent progress:","Nichol 2015; Nguyen 2023"),
        ("Quantum role:","GaAs/AlGaAs 2DEG spin qubits"),
        ("","IR single-photon emitters"),
    ])
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_S10_gaas.png",dpi=150,bbox_inches="tight"); plt.close()

# ====================================================================
# S11 - InAs (zincblende)
# ====================================================================
def fig_s11_inas():
    fig = plt.figure(figsize=(9,4.2))
    ax1 = fig.add_subplot(121,projection="3d")
    A,B = diamond_positions(); pos=A+B
    cols = ["#8e44ad"]*len(A)+["#6aa84f"]*len(B)
    sizes = [180]*len(A)+[160]*len(B)
    draw_crystal(ax1,pos,cols,sizes,bonds=diamond_bonds(A,B),title="InAs — Zincblende")
    ax1.view_init(elev=20,azim=30)
    ax1.scatter([],[],[],c="#8e44ad",s=60,label="In"); ax1.scatter([],[],[],c="#6aa84f",s=60,label="As")
    ax1.legend(loc="upper right",fontsize=8,frameon=False)
    ax2 = fig.add_subplot(122)
    info_panel(ax2,"Indium Arsenide (InAs)",[
        ("Crystal system:","Zincblende, a = 6.06 Å"),
        ("Band gap:","0.35 eV (direct, 300 K)"),
        ("Electron eff. mass:","m* = 0.023 m_e"),
        ("Mobility (electron):","> 30 000 cm²/(V·s)"),
        ("Spin-orbit:","Strong Rashba coupling"),
        ("g-factor:","|g*| ≈ 15"),
        ("Quantum role:","InAs–Al Majorana nanowires"),
        ("","semi-SC hybrid platforms"),
    ])
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_S11_inas.png",dpi=150,bbox_inches="tight"); plt.close()

# ====================================================================
# S12 - Bi2Te3 + Dirac surface state
# ====================================================================
def fig_s12_bi2te3():
    fig = plt.figure(figsize=(9,4.2),dpi=150)
    gsL = fig.add_axes([0.05,0.08,0.38,0.84]); gsR = fig.add_axes([0.50,0.12,0.45,0.78])
    ax = gsL; ax.set_xlim(-1.2,1.2); ax.set_ylim(-0.5,7.5); ax.set_aspect("equal"); ax.axis("off")
    ql_y=[0.3,1.0,1.7,2.4,3.1, 4.1,4.8,5.5,6.2,6.9]
    atom=["Te","Bi","Te","Bi","Te","Te","Bi","Te","Bi","Te"]
    dx_=[0,0.25,0.5,0.25,0, 0,0.25,0.5,0.25,0]
    cm={"Bi":"#7d2d8f","Te":"#d4a017"}; sm={"Bi":420,"Te":320}
    for y,t,dx in zip(ql_y,atom,dx_):
        for x0 in [-0.9,-0.4,0.1,0.6]:
            ax.scatter(x0+dx,y,s=sm[t],c=cm[t],edgecolors="k",linewidths=0.6,zorder=3)
    ax.annotate("",xy=(-1.05,0.25),xytext=(-1.05,3.15),arrowprops=dict(arrowstyle="<->",color="k",lw=1.1))
    ax.text(-1.15,1.7,"QL",ha="right",va="center",fontsize=10,style="italic",rotation=90)
    ax.annotate("",xy=(-1.05,4.05),xytext=(-1.05,6.95),arrowprops=dict(arrowstyle="<->",color="k",lw=1.1))
    ax.text(-1.15,5.5,"QL",ha="right",va="center",fontsize=10,style="italic",rotation=90)
    ax.add_patch(Rectangle((-1.1,3.3),2.3,0.6,fc="none",ec="gray",ls=":",lw=0.8))
    ax.text(1.2,3.6,"van der Waals\ngap",ha="left",va="center",fontsize=8,style="italic",color="gray")
    ax.scatter([],[],s=180,c=cm["Bi"],edgecolors="k",linewidths=0.5,label="Bi")
    ax.scatter([],[],s=140,c=cm["Te"],edgecolors="k",linewidths=0.5,label="Te")
    ax.legend(loc="lower center",ncol=2,frameon=False,fontsize=9,bbox_to_anchor=(0.5,-0.05))
    ax.set_title("Bi$_2$Te$_3$ quintuple-layer structure",fontsize=10)
    ax = gsR
    k = np.linspace(-1,1,200)
    Ec = 0.15 + 0.6*k**2; Ev = -0.15 - 0.6*k**2
    Ed1 = 0.6*np.abs(k); Ed2 = -0.6*np.abs(k)
    ax.fill_between(k,Ec,1.1,color="#87ceeb",alpha=0.25,label="Bulk CB")
    ax.fill_between(k,-1.1,Ev,color="#f4a582",alpha=0.25,label="Bulk VB")
    ax.plot(k,Ec,color="#2166ac",lw=1.2); ax.plot(k,Ev,color="#b2182b",lw=1.2)
    ax.plot(k,Ed1,color="k",lw=2.0,label="Surface (Dirac)"); ax.plot(k,Ed2,color="k",lw=2.0)
    ax.axhline(0,color="gray",lw=0.6,ls="--")
    ax.annotate("",xy=(0,0.15),xytext=(0,-0.15),arrowprops=dict(arrowstyle="<->",color="gray",lw=0.9))
    ax.text(0.05,0,"E$_g$ ≈ 0.15 eV",fontsize=8,color="gray",va="center")
    ax.plot(0,0,"ko",ms=5); ax.text(0.02,0.05,"Dirac\npoint",fontsize=8,va="bottom")
    for ks,sign in [(-0.6,1),(0.6,-1)]:
        ax.annotate("",xy=(ks+0.15*sign,0.36),xytext=(ks,0.36),
                    arrowprops=dict(arrowstyle="->",color="purple",lw=1.2))
    ax.text(0,0.55,"spin-momentum locked",fontsize=8,color="purple",ha="center",style="italic")
    ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_xlabel("k (surface)")
    ax.set_ylabel("Energy E − E$_F$ (a.u.)"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Topological surface states (3D TI)",fontsize=10)
    ax.legend(loc="lower right",fontsize=7,framealpha=0.9)
    plt.savefig(f"{OUT}/fig_S12_bi2te3.png",dpi=150,bbox_inches="tight",facecolor="white"); plt.close()

# ====================================================================
# S13 - InAs/Al hybrid nanowire
# ====================================================================
def fig_s13_inas_al():
    fig = plt.figure(figsize=(9,4.2),dpi=150)
    axL = fig.add_axes([0.04,0.08,0.43,0.84]); axR = fig.add_axes([0.54,0.12,0.42,0.78])
    axL.set_xlim(-1.6,1.6); axL.set_ylim(-1.6,1.6); axL.set_aspect("equal"); axL.axis("off")
    theta = np.linspace(0,2*np.pi,7); r=0.7
    xh = r*np.cos(theta+np.pi/6); yh = r*np.sin(theta+np.pi/6)
    axL.fill(xh,yh,color="#ff8c00",alpha=0.85,edgecolor="k",linewidth=1.2)
    axL.text(0,0,"InAs\ncore",ha="center",va="center",fontsize=10,fontweight="bold")
    for i in [0,1,2]:
        p1=(xh[i],yh[i]); p2=(xh[i+1],yh[i+1])
        mid = np.array([(p1[0]+p2[0])/2,(p1[1]+p2[1])/2]); n = mid/np.linalg.norm(mid)
        out1 = np.array(p1)+0.25*n; out2 = np.array(p2)+0.25*n
        axL.fill([p1[0],p2[0],out2[0],out1[0]],[p1[1],p2[1],out2[1],out1[1]],
                 color="#a9a9a9",edgecolor="k",linewidth=0.9)
    axL.text(-1.25,0.9,"Al (SC)\nshell",fontsize=9,color="#444",fontweight="bold")
    axL.annotate("",xy=(-0.9,0.6),xytext=(-1.05,0.85),arrowprops=dict(arrowstyle="->",color="k",lw=0.9))
    axL.text(1.0,-1.1,"interface\nmust be\natomically\nclean",fontsize=8,color="#b2182b",style="italic")
    axL.annotate("",xy=(0.55,-0.45),xytext=(0.95,-0.75),arrowprops=dict(arrowstyle="->",color="#b2182b",lw=0.8))
    axL.set_title("InAs/Al hybrid nanowire (cross-section)",fontsize=10)
    axR.set_xlim(0,1); axR.set_ylim(-1.3,1.0)
    axR.add_patch(Rectangle((0.05,0.22),0.25,0.70,color="#cccccc",alpha=0.6))
    axR.add_patch(Rectangle((0.05,-0.92),0.25,0.70,color="#cccccc",alpha=0.6))
    axR.text(0.175,-1.10,"Al",ha="center",fontsize=10,fontweight="bold")
    axR.annotate("",xy=(0.175,0.22),xytext=(0.175,-0.22),arrowprops=dict(arrowstyle="<->",color="k",lw=1.0))
    axR.text(0.23,0,"Δ$_{Al}$\n≈ 0.2 meV",fontsize=8,va="center")
    axR.add_patch(Rectangle((0.40,0.05),0.15,0.87,color="#ffd9a6",alpha=0.6))
    axR.add_patch(Rectangle((0.40,-0.92),0.15,0.97,color="#ffd9a6",alpha=0.6))
    axR.text(0.475,-1.10,"InAs",ha="center",fontsize=10,fontweight="bold")
    axR.text(0.475,0,"no\ngap",fontsize=8,ha="center",color="#b2182b",style="italic")
    axR.add_patch(Rectangle((0.70,0.10),0.25,0.82,color="#ffd9a6",alpha=0.6))
    axR.add_patch(Rectangle((0.70,-0.92),0.25,1.02,color="#ffd9a6",alpha=0.6))
    axR.text(0.825,-1.10,"InAs + Al shell",ha="center",fontsize=10,fontweight="bold")
    axR.annotate("",xy=(0.825,0.10),xytext=(0.825,-0.10),arrowprops=dict(arrowstyle="<->",color="#2166ac",lw=1.1))
    axR.text(0.88,0,"Δ*\ninduced",fontsize=8,va="center",color="#2166ac")
    axR.set_xticks([]); axR.set_yticks([]); axR.set_ylabel("Energy",fontsize=9)
    axR.set_title("Superconducting proximity effect",fontsize=10,pad=10)
    for s in ["top","right","bottom"]: axR.spines[s].set_visible(False)
    plt.savefig(f"{OUT}/fig_S13_inas_al.png",dpi=150,bbox_inches="tight",facecolor="white"); plt.close()

# ====================================================================
# S14 - Diamond NV- centre
#   Level scheme drawn approximately to scale (y = 0.12 + E/eV * 0.324):
#   3A2 = 0, 1E ~ 0.34 eV, 1A1 ~ 1.51 eV, 3E = 1.945 eV (637 nm ZPL).
#   Correct singlet ordering: 1A1 (upper) -> 1042 nm -> 1E (lower).
#   ISC: 3E -> 1A1 (spin-selective), then 1E -> 3A2 (m_s = 0 preferred).
# ====================================================================
def fig_s14_diamond_nv():
    fig = plt.figure(figsize=(9,4.5),dpi=150)
    axL = fig.add_axes([0.02,0.05,0.40,0.90],projection="3d")
    axR = fig.add_axes([0.52,0.08,0.44,0.86])
    N_pos = np.array([[0.0,0.0,0.0]]); Vac = np.array([[0.5,0.5,0.5]])
    C_sites = np.array([[1.0,1.0,0.0],[1.0,0.0,1.0],[0.0,1.0,1.0],[1.0,0.0,0.0],
                        [0.0,1.0,0.0],[0.0,0.0,1.0],[1.0,1.0,1.0],[0.5,0.0,0.5],
                        [0.5,0.5,0.0],[0.0,0.5,0.5]])
    axL.scatter(C_sites[:,0],C_sites[:,1],C_sites[:,2],s=200,c="#333333",
                edgecolors="k",linewidths=0.5,depthshade=True,label="C")
    axL.scatter(N_pos[:,0],N_pos[:,1],N_pos[:,2],s=260,c="#3176c9",
                edgecolors="k",linewidths=0.8,label="N")
    axL.scatter(Vac[:,0],Vac[:,1],Vac[:,2],s=260,facecolors="none",edgecolors="red",
                linewidths=1.5,linestyle="--",label="V (vacancy)")
    def bond(p1,p2,color="k",lw=1.0,ls="-"):
        axL.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],color=color,lw=lw,ls=ls)
    for p in [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]: bond(N_pos[0],p,color="#3176c9",lw=1.4)
    for p in [[1.0,1.0,0.0],[1.0,0.0,1.0],[0.0,1.0,1.0]]: bond(Vac[0],p,color="red",lw=1.0,ls="--")
    bond(N_pos[0],Vac[0],color="purple",lw=2.2)
    axL.text(0.05,0.05,0.35,"NV axis",color="purple",fontsize=9,fontweight="bold")
    axL.set_xlim(0,1); axL.set_ylim(0,1); axL.set_zlim(0,1)
    axL.set_xticks([]); axL.set_yticks([]); axL.set_zticks([])
    axL.set_title("NV$^-$ centre in diamond",fontsize=10)
    axL.legend(loc="upper left",fontsize=7,framealpha=0.9); axL.view_init(elev=18,azim=35)

    # --- level scheme ---------------------------------------------------
    y3A2, y3E, y1A1, y1E = 0.12, 0.75, 0.61, 0.23
    axR.set_xlim(0,1); axR.set_ylim(0,1); axR.axis("off")
    # ground triplet 3A2 (m_s = 0 and m_s = +/-1, zero-field splitting exaggerated)
    axR.hlines(y3A2,0.1,0.45,colors="k",lw=2); axR.text(0.08,y3A2,"$^3A_2$",ha="right",va="center",fontsize=10)
    axR.hlines(0.30,0.1,0.45,colors="k",lw=2)
    axR.text(0.47,y3A2,"$m_s=0$",fontsize=8,va="center")
    axR.text(0.47,0.30,"$m_s=\\pm 1$",fontsize=8,va="center")
    axR.annotate("",xy=(0.28,0.30),xytext=(0.28,y3A2),arrowprops=dict(arrowstyle="<->",color="darkgreen",lw=1.1))
    axR.text(0.30,0.21,"D ≈ 2.87 GHz",fontsize=8,color="darkgreen",va="center")
    # excited triplet 3E
    axR.hlines(y3E,0.1,0.45,colors="k",lw=2); axR.text(0.08,y3E,"$^3E$",ha="right",va="center",fontsize=10)
    axR.hlines(y3E+0.05,0.1,0.45,colors="k",lw=2)
    axR.annotate("",xy=(0.22,y3E),xytext=(0.22,y3A2),arrowprops=dict(arrowstyle="->",color="#b2182b",lw=1.3))
    axR.annotate("",xy=(0.18,y3A2+0.08),xytext=(0.18,y3E-0.01),arrowprops=dict(arrowstyle="->",color="#b2182b",lw=1.0,ls="--"))
    axR.text(0.02,0.45,"637 nm\nZPL",fontsize=8,color="#b2182b",ha="left")
    # singlets: 1A1 above 1E
    axR.hlines(y1A1,0.58,0.85,colors="gray",lw=2); axR.text(0.87,y1A1,"$^1A_1$",fontsize=10,va="center")
    axR.hlines(y1E,0.58,0.85,colors="gray",lw=2);  axR.text(0.87,y1E,"$^1E$",fontsize=10,va="center")
    axR.annotate("",xy=(0.72,y1E),xytext=(0.72,y1A1),arrowprops=dict(arrowstyle="->",color="gray",lw=0.9))
    axR.text(0.74,(y1A1+y1E)/2,"1042 nm",fontsize=7,color="gray",va="center")
    # intersystem crossing 3E -> 1A1, then 1E -> 3A2 (m_s = 0)
    axR.annotate("",xy=(0.58,y1A1),xytext=(0.46,y3E+0.03),
                 arrowprops=dict(arrowstyle="->",color="orange",lw=1.1,connectionstyle="arc3,rad=-0.2"))
    axR.text(0.47,0.83,"ISC (spin-selective)",fontsize=7,color="orange")
    axR.annotate("",xy=(0.45,y3A2),xytext=(0.58,y1E),
                 arrowprops=dict(arrowstyle="->",color="orange",lw=1.0,ls="--",connectionstyle="arc3,rad=-0.2"))
    axR.annotate("",xy=(0.40,0.30),xytext=(0.40,y3A2),arrowprops=dict(arrowstyle="<->",color="#2166ac",lw=1.2))
    axR.text(0.40,0.05,"MW drive",fontsize=7.5,color="#2166ac",ha="center")
    axR.set_title("NV$^-$ electronic structure",fontsize=10,pad=8)
    axR.text(0.5,-0.02,"S=1 triplet · room-T T$_2$ ~ ms · D-W ≈ 0.03",fontsize=7.5,
             ha="center",style="italic",color="#444")
    plt.savefig(f"{OUT}/fig_S14_diamond_nv.png",dpi=150,bbox_inches="tight",facecolor="white"); plt.close()

# ====================================================================
# S15 - Si3N4 / SiO2 photonic platforms
# ====================================================================
def fig_s15_sin_sio2():
    fig = plt.figure(figsize=(9,4.2),dpi=150)
    axL = fig.add_axes([0.04,0.10,0.42,0.80]); axR = fig.add_axes([0.54,0.10,0.42,0.80])
    axL.set_xlim(0,8); axL.set_ylim(0,5); axL.set_aspect("equal"); axL.axis("off")
    axL.add_patch(Rectangle((0.5,0.3),7,1.0,color="#888888",edgecolor="k"))
    axL.text(4,0.8,"Si substrate",ha="center",va="center",fontsize=9,color="white",fontweight="bold")
    axL.add_patch(Rectangle((0.5,1.3),7,1.3,color="#d9d9f3",edgecolor="k"))
    axL.text(1.5,1.95,"SiO$_2$   n ≈ 1.45",ha="left",va="center",fontsize=9)
    axL.add_patch(Rectangle((3.2,2.6),1.6,0.7,color="#5ab4ac",edgecolor="k"))
    axL.text(4.0,2.95,"Si$_3$N$_4$",ha="center",va="center",fontsize=10,fontweight="bold")
    axL.annotate("n ≈ 2.0",xy=(4.85,2.95),xytext=(5.6,3.3),
                 arrowprops=dict(arrowstyle="->",color="k",lw=0.8),fontsize=9)
    axL.add_patch(Rectangle((0.5,3.3),7,1.0,color="#d9d9f3",alpha=0.6,edgecolor="k"))
    axL.text(1.5,3.8,"SiO$_2$ (top-cladding)",ha="left",va="center",fontsize=9)
    mode_x = np.linspace(3.0,5.0,50)
    mode_top = 3.3 + 0.4*np.exp(-((mode_x-4.0)/0.6)**2)
    mode_bot = 2.55 - 0.3*np.exp(-((mode_x-4.0)/0.6)**2)
    axL.fill_between(mode_x,mode_bot,mode_top,color="red",alpha=0.18)
    axL.contour(np.linspace(2.5,5.5,60),np.linspace(2.3,3.8,40),
                np.exp(-((np.linspace(2.5,5.5,60)[None,:]-4)/0.6)**2
                       -((np.linspace(2.3,3.8,40)[:,None]-2.95)/0.3)**2),
                levels=[0.3,0.6,0.9],colors="red",alpha=0.6,linewidths=0.9)
    axL.text(6.2,2.95,"TE mode",fontsize=8,color="red",style="italic")
    axL.annotate("",xy=(5.3,2.95),xytext=(6.1,2.95),arrowprops=dict(arrowstyle="->",color="red",lw=0.8))
    axL.set_title("Si$_3$N$_4$ strip waveguide\n(loss < 0.1 dB/m, CMOS-compatible)",fontsize=10)
    axR.set_xlim(-2.6,2.6); axR.set_ylim(-2.6,2.6); axR.set_aspect("equal"); axR.axis("off")
    axR.add_patch(Circle((0,0),2.2,color="#d9d9f3",ec="k"))
    axR.text(1.55,1.55,"SiO$_2$\ncladding\nn ≈ 1.44",ha="left",va="bottom",fontsize=8)
    axR.add_patch(Circle((0,0),0.55,color="#f6ad55",ec="k"))
    axR.text(0,0,"GeO$_2$-\ndoped\ncore",ha="center",va="center",fontsize=8,fontweight="bold")
    axR.annotate("n ≈ 1.46",xy=(0.4,0.4),xytext=(1.1,1.0),
                 arrowprops=dict(arrowstyle="->",color="k",lw=0.7),fontsize=8)
    th = np.linspace(0,2*np.pi,60)
    axR.fill(0.7*np.cos(th),0.7*np.sin(th),color="red",alpha=0.12)
    axR.set_title("SiO$_2$ optical fibre\n(loss < 0.2 dB/km at 1550 nm)",fontsize=10)
    fig.text(0.5,0.02,"Refractive indices: Si (3.48) > Si$_3$N$_4$ (2.0) > SiO$_2$ (1.45)",
             ha="center",fontsize=8.5,style="italic",color="#333")
    plt.savefig(f"{OUT}/fig_S15_sin_sio2.png",dpi=150,bbox_inches="tight",facecolor="white"); plt.close()

# ====================================================================
# S20 - Stress-strain: hBN monolayer vs steel
#   hBN: E ≈ 0.865 TPa, fracture strength ≈ 70 GPa (Falin et al. 2017)
# ====================================================================
def fig_s20_mech():
    fig,ax = plt.subplots(figsize=(7.5,4.5))
    eps = np.linspace(0,0.08,200)
    def stress(eps,E,smax):
        s = E*eps; return np.where(s<smax,s,smax)
    s_hBN = stress(eps,865.0,70.0); s_steel = stress(eps,200.0,0.8)
    ax.plot(eps*100,s_hBN,lw=2.5,color="#2874a6",
            label="hBN monolayer: E ≈ 0.87 TPa, σ_max ≈ 70 GPa")
    ax.plot(eps*100,s_steel,lw=2.5,color="#c0392b",
            label="Structural steel: E ≈ 200 GPa, σ_max ≈ 0.8 GPa")
    ax.axhline(70,color="#2874a6",ls=":",alpha=0.7); ax.axhline(0.8,color="#c0392b",ls=":",alpha=0.7)
    ax.set_xlabel("Strain ε (%)"); ax.set_ylabel("Stress σ (GPa)")
    ax.set_title("Fig. S20 — Young's modulus = slope ≠ tensile strength σ_max")
    ax.legend(loc="upper right",fontsize=9); ax.grid(alpha=0.3,which="both")
    ax.set_yscale("log"); ax.set_ylim(0.1,1500)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_S20_mech.png",dpi=160); plt.close()

# ====================================================================
# S21 - V_B- defect in the hBN honeycomb lattice
#   Proper honeycomb: Bravais vectors a1 = (a, 0), a2 = (a/2, a*sqrt(3)/2),
#   basis B at (0, 0) and N at (0, d) with d = a/sqrt(3) (B-N bond length).
#   The boron atom at the origin is removed; its three N neighbours keep
#   dangling bonds (drawn in red).
# ====================================================================
def fig_s21_vb_hbn():
    fig,ax = plt.subplots(figsize=(7,6))
    a = 1.0; d = a/np.sqrt(3)
    a1 = np.array([a,0.0]); a2 = np.array([a/2, a*np.sqrt(3)/2])
    XL, YL = 4.0, 3.4

    def inside(p): return abs(p[0]) <= XL and abs(p[1]) <= YL

    B_sites, N_sites = [], []
    for i in range(-8,9):
        for j in range(-8,9):
            R = i*a1 + j*a2
            if inside(R): B_sites.append(R)
            Nn = R + np.array([0.0,d])
            if inside(Nn): N_sites.append(Nn)

    nn_vecs = [np.array([0.0,d]), np.array([a/2,-d/2]), np.array([-a/2,-d/2])]
    vac = np.array([0.0,0.0])
    vac_neighbours = [vac+v for v in nn_vecs]

    # bonds (each B to its three N neighbours)
    for b in B_sites:
        is_vac = np.linalg.norm(b-vac) < 1e-6
        for v in nn_vecs:
            n = b+v
            if not inside(n): continue
            if is_vac:
                ax.plot([b[0]+0.45*v[0],n[0]],[b[1]+0.45*v[1],n[1]],
                        color="red",lw=1.4,ls=":",zorder=1)
            else:
                ax.plot([b[0],n[0]],[b[1],n[1]],color="gray",lw=1.0,zorder=1)

    # atoms
    for b in B_sites:
        if np.linalg.norm(b-vac) < 1e-6:
            ax.add_patch(Circle(b,0.22,fill=False,ec="red",lw=2.2,ls="--",zorder=3))
            ax.text(b[0],b[1],r"$V_B^-$",ha="center",va="center",color="red",
                    fontsize=10,fontweight="bold",zorder=4)
            continue
        ax.add_patch(Circle(b,0.17,color="#3498db",ec="black",lw=0.8,zorder=2))
        ax.text(b[0],b[1],"B",ha="center",va="center",color="white",fontsize=8,fontweight="bold",zorder=3)
    for n in N_sites:
        near_vac = any(np.linalg.norm(n-m) < 1e-6 for m in vac_neighbours)
        ax.add_patch(Circle(n,0.17,color="#e67e22",ec="red" if near_vac else "black",
                            lw=2.0 if near_vac else 0.8,zorder=2))
        ax.text(n[0],n[1],"N",ha="center",va="center",color="white",fontsize=8,fontweight="bold",zorder=3)

    ax.set_xlim(-XL-0.3,XL+0.3); ax.set_ylim(-YL-0.3,YL+0.3)
    ax.set_aspect("equal"); ax.axis("off")
    legend = [
        mpatches.Patch(color="#3498db",label="Boron (B)"),
        mpatches.Patch(color="#e67e22",label="Nitrogen (N)"),
        Line2D([0],[0],marker="o",markerfacecolor="none",markeredgecolor="red",
               linestyle="--",markersize=12,label=r"$V_B^-$ vacancy (S = 1, ZFS ≈ 3.48 GHz)"),
        Line2D([0],[0],color="red",ls=":",lw=1.4,label="Dangling bonds of the 3 N neighbours"),
    ]
    ax.legend(handles=legend,loc="lower center",bbox_to_anchor=(0.5,-0.14),
              ncol=1,fontsize=9,frameon=False)
    ax.set_title(r"Fig. S21 — Negatively charged boron vacancy $V_B^-$ in monolayer hBN")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_S21_vb_hbn.png",dpi=160,bbox_inches="tight"); plt.close()


# ====================================================================
# M1 - Coherence time vs material strategy (schematic synthesis)
# ====================================================================
def fig_m1_coherence():
    fig,ax = plt.subplots(figsize=(9,5))
    rows = [
        ("Transmon — natural Al/AlO_x, sapphire (2007)",1e-6,C_SC,"Koch 2007"),
        ("Transmon — Nb electrodes, Si substrate (~2015)",1.5e-5,C_SC,"Nb / Nb₂O₅"),
        ("Transmon — Ta on sapphire (Place 2021)",3e-4,C_SC,"Ta / Ta₂O₅"),
        ("Transmon — Nb-encapsulated 2D (Bland 2025)",1.2e-3,C_SC,"Nb surface eng."),
        ("Si spin qubit — natural Si (pre-2010)",3e-6,C_SPIN,"4.7% ²⁹Si bath"),
        ("Si QD spin qubit — ²⁸Si, Hahn echo (Veldhorst 2014)",2.8e-2,C_SPIN,"isotopic enrichment"),
        ("Ge hole qubit — Ge/SiGe (Jirovec 2021)",1e-4,C_SPIN,"spin-orbit, low ⁷³Ge"),
        ("V_B⁻ in hBN — natural (Gottscholl 2020)",1e-7,C_DEFECT,"native h^{nat}BN"),
        ("V_B⁻ in hBN — DD pulses (Haykal 2022)",1e-4,C_DEFECT,"dynamical decoupling"),
        ("NV⁻ in diamond — ¹²C-enriched (Bar-Gill 2013)",2e-3,C_DEFECT,"¹²C enrichment"),
        ("InAs/Al hybrid nanowire (Aghaee 2025)",1e-6,C_TI,"proximity-induced Δ*"),
    ]
    y = np.arange(len(rows))[::-1]
    T2 = [r[1] for r in rows]; cols = [r[2] for r in rows]
    labels = [r[0] for r in rows]; notes = [r[3] for r in rows]
    ax.barh(y,T2,color=cols,edgecolor="black",linewidth=0.6,alpha=0.85)
    for yi,t,n in zip(y,T2,notes):
        ax.text(t*1.25,yi,n,va="center",fontsize=8.5,style="italic",color="#444")
    ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=9)
    ax.set_xscale("log"); ax.set_xlim(1e-8,1e0)
    ax.set_xlabel("Coherence time T₂ (s) — log scale, order of magnitude")
    ax.set_title("Fig. M1 — Coherence time across platforms set by the underlying material strategy",
                 fontsize=11,pad=10)
    for x,lab in [(1e-6,"1 µs"),(1e-3,"1 ms"),(1e0,"1 s")]:
        ax.axvline(x,color=C_GRID,lw=0.7,ls="--",zorder=0)
        ax.text(x,len(rows)-0.3,lab,ha="center",va="bottom",fontsize=8,color="#888")
    leg = [Patch(facecolor=C_SC,edgecolor="k",label="Superconducting"),
           Patch(facecolor=C_SPIN,edgecolor="k",label="Semiconductor spin"),
           Patch(facecolor=C_DEFECT,edgecolor="k",label="Solid-state defect"),
           Patch(facecolor=C_TI,edgecolor="k",label="Topological / hybrid")]
    ax.legend(handles=leg,loc="lower right",fontsize=8.5,frameon=True)
    ax.grid(axis="x",alpha=0.25,which="both")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_M1_coherence.png",dpi=160,bbox_inches="tight"); plt.close()

# ====================================================================
# M2 - 1/f TLS noise (amplitude at 1 Hz set equal to the quoted tan δ)
# ====================================================================
def fig_m2_noise():
    fig,ax = plt.subplots(figsize=(8,4.8))
    f = np.logspace(0,9,400)
    def S(f,A,alpha=1.0): return A/f**alpha
    ax.loglog(f,S(f,1e-2),color="#c0392b",lw=2.2,label="Amorphous AlO_x JJ barrier  (tan δ ≈ 10⁻²)")
    ax.loglog(f,S(f,1e-3),color="#d35400",lw=2.0,label="Nb₂O₅ surface oxide        (tan δ ≈ 10⁻³)")
    ax.loglog(f,S(f,1e-4),color="#16a085",lw=2.0,label="SiO₂ / Si interface         (tan δ ≈ 10⁻⁴)")
    ax.loglog(f,S(f,1e-5),color="#2874a6",lw=2.2,label="Ta₂O₅ on sapphire        (tan δ ≈ 10⁻⁵)")
    ax.loglog(f,S(f,1e-6),color="#27ae60",lw=2.0,label="Diamond / NV⁻ surface     (tan δ ≈ 10⁻⁶)")
    ax.axvspan(1e3,1e7,color="#fff2cc",alpha=0.4,zorder=0)
    ax.text(1e5,3e-4,"qubit-relevant\nband (kHz–MHz)",ha="center",va="center",
            fontsize=9,style="italic",color="#7d6608")
    ax.text(3e1,3e-5,r"$S(f) \propto 1/f$",fontsize=11,color="#444")
    ax.set_xlabel("Frequency f (Hz)"); ax.set_ylabel("Noise spectral density S(f) (arb. units)")
    ax.set_title("Fig. M2 — Two-level-system 1/f noise spectrum across material interfaces",
                 fontsize=11,pad=10)
    ax.grid(alpha=0.3,which="both"); ax.legend(loc="upper right",fontsize=8.5,framealpha=0.95)
    ax.set_xlim(1,1e9); ax.set_ylim(1e-16,1e-1)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_M2_noise.png",dpi=160,bbox_inches="tight"); plt.close()

# ====================================================================
# M3 - Gate-error progression vs threshold (schematic synthesis)
# ====================================================================
def fig_m3_error_rates():
    fig,ax = plt.subplots(figsize=(8.5,5))
    sc_1q = [(2009,5e-2),(2012,8e-3),(2014,2e-3),(2017,6e-4),(2019,3e-4),(2021,1.5e-4),(2024,8e-5)]
    sc_2q = [(2014,5e-2),(2017,1e-2),(2019,5e-3),(2021,2.5e-3),(2023,8e-4),(2024,5e-4)]
    si_2q = [(2015,1e-1),(2018,3e-2),(2020,8e-3),(2022,2e-3),(2024,1e-3)]
    ion_2q = [(2014,1e-3),(2016,6e-4),(2019,2e-4),(2021,1e-4),(2024,5e-5)]
    def plot(series,color,marker,label):
        yrs=[s[0] for s in series]; err=[s[1] for s in series]
        ax.semilogy(yrs,err,marker=marker,color=color,lw=2,markersize=8,
                    label=label,markeredgecolor="black",markeredgewidth=0.5)
    plot(sc_1q,"#c0392b","o","Superconducting — 1-qubit")
    plot(sc_2q,"#e67e22","s","Superconducting — 2-qubit")
    plot(si_2q,"#2874a6","D","Si spin — 2-qubit")
    plot(ion_2q,"#8e44ad","^","Trapped ion — 2-qubit")
    ax.axhspan(5e-3,1e-2,color="#ffe5cc",alpha=0.55,zorder=0)
    ax.axhline(1e-2,color=C_TARGET,ls="--",lw=1.4)
    ax.text(2009.5,1.1e-2,"surface-code threshold ≈ 1 %",fontsize=9,color=C_TARGET,fontweight="bold")
    ax.axhline(1e-4,color="#117a65",ls=":",lw=1.4)
    ax.text(2009.5,1.1e-4,"fault-tolerant target ≈ 10⁻⁴",fontsize=9,color="#117a65",fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Gate error rate ε (per gate) — log scale")
    ax.set_title("Fig. M3 — Gate-fidelity progression — material engineering drives error suppression",
                 fontsize=11,pad=10)
    ax.legend(loc="lower left",fontsize=9,framealpha=0.95)
    ax.grid(alpha=0.3,which="both"); ax.set_xlim(2008.5,2025.5); ax.set_ylim(1e-5,2e-1)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_M3_error_rates.png",dpi=160,bbox_inches="tight"); plt.close()

# ====================================================================
# M4 - Scalability (schematic synthesis)
#   NOTE: panel (a) values are illustrative; verify against the
#   original sources before reuse (see README).
# ====================================================================
def fig_m4_scalability():
    fig,axes = plt.subplots(1,2,figsize=(11,4.6))
    ax = axes[0]
    methods = [("Manual\nshadow-evap.\nAl/AlO_x (2005)",45),
               ("Optimized\nAl/AlO_x (2015)",18),
               ("Targeted\nlaser annealing\n(2021)",5),
               ("AI-corrected\nprocess (2024)",2),
               ("Ta/sapphire\nencapsulated\n(2025)",1.2)]
    labels=[m[0] for m in methods]; sigma=[m[1] for m in methods]
    xs = np.arange(len(methods))
    ax.bar(xs,sigma,color=[C_SC]*len(methods),edgecolor="black",linewidth=0.7,alpha=0.85)
    ax.axhline(1,color=C_TARGET,ls="--",lw=1.4)
    ax.text(0.05,1.3,"uniformity required for\nfault-tolerant 1000-qubit chip  (≤ 1%)",
            fontsize=8.5,color=C_TARGET,fontweight="bold")
    for x,s in zip(xs,sigma):
        ax.text(x,s+0.6,f"{s}%",ha="center",fontsize=9,fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(labels,fontsize=8)
    ax.set_ylabel("Josephson-junction frequency spread σ/f (%)")
    ax.set_yscale("log"); ax.set_ylim(0.5,80)
    ax.set_title("(a) Material processing reduces JJ disorder",fontsize=10.5,pad=8)
    ax.grid(axis="y",alpha=0.3,which="both")
    ax2 = axes[1]
    sc=[(2007,1),(2014,9),(2017,17),(2019,53),(2021,127),(2022,433),(2023,1121)]
    spin=[(2012,1),(2016,2),(2018,4),(2020,6),(2022,6),(2024,16)]
    ion=[(2010,4),(2015,7),(2018,11),(2020,16),(2022,32),(2024,56)]
    neutral=[(2016,18),(2018,51),(2020,196),(2022,256),(2023,1180)]
    for series,color,marker,label in [
        (sc,"#c0392b","o","Superconducting"),
        (spin,"#2874a6","D","Si/Ge spin"),
        (ion,"#8e44ad","^","Trapped ion"),
        (neutral,"#16a085","s","Neutral-atom array")]:
        yrs=[s[0] for s in series]; n=[s[1] for s in series]
        ax2.semilogy(yrs,n,marker=marker,color=color,lw=2,markersize=7,
                     label=label,markeredgecolor="k",markeredgewidth=0.4)
    ax2.set_xlabel("Year"); ax2.set_ylabel("Physical qubit count (log)")
    ax2.set_xlim(2006.5,2025.5); ax2.set_ylim(1,5000)
    ax2.grid(alpha=0.3,which="both"); ax2.legend(loc="upper left",fontsize=8.5,framealpha=0.95)
    ax2.set_title("(b) Scaling rate set by which interfaces dominate fabrication",fontsize=10.5,pad=8)
    fig.suptitle("Fig. M4 — Scalability: material uniformity and qubit count",fontsize=11,y=1.02)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_M4_scalability.png",dpi=160,bbox_inches="tight"); plt.close()


# ====================================================================
# Run everything
# ====================================================================
ALL = [
    ("S1  transmon",           fig_s1_transmon),
    ("S2  surface code",       fig_s2_threshold),
    ("S3  Zeeman",             fig_s3_zeeman),
    ("S4  Al (FCC)",           fig_s4_al),
    ("S5  Nb (BCC)",           fig_s5_nb),
    ("S6  NbN (rocksalt)",     fig_s6_nbn),
    ("S7  TiN (rocksalt)",     fig_s7_tin),
    ("S8  Si (diamond)",       fig_s8_si),
    ("S9  Ge (diamond)",       fig_s9_ge),
    ("S10 GaAs (zincblende)",  fig_s10_gaas),
    ("S11 InAs (zincblende)",  fig_s11_inas),
    ("S12 Bi2Te3 + Dirac",     fig_s12_bi2te3),
    ("S13 InAs/Al hybrid",     fig_s13_inas_al),
    ("S14 diamond NV-",        fig_s14_diamond_nv),
    ("S15 Si3N4 / SiO2",       fig_s15_sin_sio2),
    ("S20 stress-strain",      fig_s20_mech),
    ("S21 V_B- in hBN",        fig_s21_vb_hbn),
    ("M1  coherence time",     fig_m1_coherence),
    ("M2  1/f noise",          fig_m2_noise),
    ("M3  error rates",        fig_m3_error_rates),
    ("M4  scalability",        fig_m4_scalability),
]

if __name__ == "__main__":
    print("Writing figures to", os.path.abspath(OUT))
    for name, fn in ALL:
        fn(); print(f"  {name:<24} OK")
    print(f"\nDone: {len(ALL)} figures.")
