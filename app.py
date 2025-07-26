# Gas‑Condensate MFHW Simulator – 3‑Phase (clean build)
# -------------------------------------------------------------
# Author: ChatGPT  |  26‑Jul‑2025
# -------------------------------------------------------------
"""
Full Streamlit prototype for a three‑phase (gas / condensate‑oil / water)
multi‑fractured horizontal‑well (MFHW) in an ultra‑low‑permeability shale.

Key physics implemented
-----------------------
• Dual‑porosity 1‑D grid   : SRV cell + matrix cell  
• Stress‑dependent k      : Yilmaz & Nur  
• Rel‑perm                : Corey (2‑p) + Stone I (3‑p)  
• PVT correlations        : Standing Rₛ, Beggs‑Robinson μₒ, BWR B_g, Carr μ_g  
• CSV smart‑units parser  : auto‑detect kPa, MMscf/d, bbl/d, etc.  
• History plot &forecast  : gas, condensate, water, BHP + EUR & CSV export

Limitations
-----------
• 2‑cell explicit solver for speed; upgrade to multi‑cell implicit for serious work  
• Live‑oil viscosity placeholder (μₒ = 1.5 cP)  
• No flash CGR decline yet – uses constant CGR_i
"""
# =============================================================
# 1  Imports & page config
# =============================================================
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="3‑Phase MFHW Simulator", layout="wide")

# =============================================================
# 2  Correlations (simplified forms)
# =============================================================

def standing_rs(api, p, t):
    return 0.00091 * api * (p ** 1.0937) * np.exp((1.25 * api) / (t + 460))

def beggs_robinson_mu_o(rs, t, api):
    a = 10 ** (3.0324 - 0.02023 * api)
    b = 10 ** (1.163 - 0.0009 * api)
    mu_dead = a * (t ** -b)
    return mu_dead * (0.68 + 0.25 * (rs/1000) + 0.062 * (rs/1000)**2)

def bwr_bg(p, t, gg):
    return 0.005 * gg * (t + 460) / p

def carr_mu_g(p, t, gg):
    return 0.0178 * np.sqrt(t) * gg / (p ** 0.1)

def stress_perm(k0, p, p_ref, mod):
    return k0 * np.exp(-mod * (p_ref - p))

# Corey & Stone I

def corey(s, s_ir, s_res, n):
    se = np.clip((s - s_ir) / (1 - s_ir - s_res), 0, 1)
    return se ** n

def stone1(sw, sg, prm):
    krw = corey(sw, prm['swirr'], prm['sorw'], prm['nw'])
    krg = corey(sg, prm['sgc'], prm['sorg'], prm['ng'])
    kro_w = corey(1 - sw, prm['sorw'], prm['swirr'], prm['no_w'])
    kro_g = corey(1 - sg, prm['sorg'], prm['sgc'], prm['no_g'])
    kro = kro_w * kro_g / (kro_w + kro_g + 1e-9)
    return krw, kro, krg

# =============================================================
# 3  PVT table builder
# =============================================================

def build_pvt(fluid, inp):
    p = np.linspace(inp['p_min'], inp['p_max'], 30)
    t = inp['T']
    tbl = {}
    tbl['gas'] = pd.DataFrame({
        'P': p,
        'Bg': bwr_bg(p, t, inp['gg']),
        'mu_g': carr_mu_g(p, t, inp['gg'])
    })
    if fluid == 'Black/Volatile Oil':
        rs = standing_rs(inp['API'], p, t)
        tbl['oil'] = pd.DataFrame({'P': p, 'Rs': rs, 'mu_o': beggs_robinson_mu_o(rs, t, inp['API'])})
    tbl['water'] = pd.DataFrame({'mu_w': [0.5], 'Bw': [1.03]})
    return tbl

# Short interpolator
def pvt_interp(p, col, table):
    return float(np.interp(p, table['P'], table[col]))

# =============================================================
# 4  Dual‑porosity explicit solver (2‑cell)
# =============================================================

def simulate(p_init, bhp, prm, pvt, rel):
    n = len(bhp)
    # state arrays
    p_srv = np.full(n, p_init)
    p_mat = np.full(n, p_init)
    sw = np.full(n, prm['Sw'])
    sg = np.full(n, prm['Sg'])
    qg = np.zeros(n); qc = np.zeros(n); qw = np.zeros(n)

    for i in range(1, n):
        # fluid props at SRV pressure
        mu_g = pvt_interp(p_srv[i-1], 'mu_g', pvt['gas'])
        Bg   = pvt_interp(p_srv[i-1], 'Bg',  pvt['gas'])
        mu_w = pvt['water']['mu_w'].iloc[0]
        mu_o = 1.5  # placeholder

        krw, kro, krg = stone1(sw[i-1], sg[i-1], rel)
        lam_w = krw/mu_w; lam_o = kro/mu_o; lam_g = krg/mu_g
        lam_t = lam_w + lam_o + lam_g
        fw, fo, fg = lam_w/lam_t, lam_o/lam_t, lam_g/lam_t

        k_srv = stress_perm(prm['k_srv'], p_srv[i-1], p_init, prm['k_mod'])
        k_mat = stress_perm(prm['k_mat'], p_mat[i-1], p_init, prm['k_mod'])
        h, rw, re = prm['h_frac'], prm['r_w'], prm['dx']
        Tw = (2*np.pi*k_srv*h)/(mu_g*np.log(re/rw))
        Tm = (2*np.pi*k_mat*h)/(mu_g*np.log(re/rw))

        q_total = Tw * (p_srv[i-1] - bhp[i])
        q_w = fw*q_total; q_o = fo*q_total; q_gas = fg*q_total
        q_matrix = Tm * (p_mat[i-1] - p_srv[i-1])

        # pressure update explicit
        dp_srv = -(q_total - q_matrix) / (prm['pv_srv']*prm['ct'])
        dp_mat = -(-q_matrix) / (prm['pv_mat']*prm['ct'])
        p_srv[i] = p_srv[i-1] + dp_srv
        p_mat[i] = p_mat[i-1] + dp_mat

        # saturation update (explicit fractional flow, 1‑day dt)
        sw[i] = np.clip(sw[i-1] + (q_w / Bg) / prm['pv_srv'], 0, 1)
        sg[i] = np.clip(sg[i-1] + (q_gas / Bg) / prm['pv_srv'], 0, 1)

        qg[i] = q_gas / Bg      # Mscf/d
        qc[i] = prm['cgr_i'] * qg[i] / 1e6
        qw[i] = q_w / Bg

    return pd.DataFrame({'day': np.arange(n), 'p_srv': p_srv,
                         'q_gas_sim': qg, 'q_cond_sim': qc, 'q_water_sim': qw})

# =============================================================
# 5  Streamlit UI & controller
# =============================================================

def std_header(col):
    name = col.lower()
    if 'gas' in name:   return 'q_gas', 1e3 if 'mm' in name else 1
    if 'cond' in name or 'oil' in name: return 'q_cond', 1
    if 'bhp' in name or 'sandface' in name or 'pressure' in name:
        return 'bhp', 0.145038 if 'kpa' in name else 1
    if 'day' in name or 'date' in name: return 'day', 1
    return None, None

def main():
    st.title("3‑Phase MFHW Simulator")

    # ---------------- Base inputs ----------------
    fluid = st.sidebar.selectbox("Fluid", ["Dry Gas","Condensate","Black/Volatile Oil"])
    p_res = st.sidebar.number_input("P_init psi", 1000., 15000., 9500.)
    T_res = st.sidebar.number_input("T_res °F", 60., 350., 250.)
    ct    = st.sidebar.number_input("ct 1/psi", 1e-6, 1e-4, 5e-6, format="%e")
    Sw = st.sidebar.number_input("Sw %", 0., 60., 20.)/100
    Sg = st.sidebar.number_input("Sg %", 0., 90., 65.)/100
    gg = st.sidebar.number_input("Gas grav", 0.55, 1.2, 0.8)

    h_frac = st.sidebar.number
