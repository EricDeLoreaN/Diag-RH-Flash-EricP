import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Audit Social (Version Robuste)", layout="wide")

st.title("📊 Analyse Sociale & Absentéisme")
st.markdown("""
Cet outil analyse vos données RH (Format ARACT ou CDC).
Il accepte les fichiers Excel (.xlsx) et CSV (.csv).
""")

# --- FONCTION DE CHARGEMENT BLINDÉE ---
@st.cache_data
def load_data(file):
    # Cette fonction essaie toutes les méthodes de lecture possibles
    # jusqu'à ce que ça marche.
    
    # 1. Tentative : Lire comme un Excel standard
    try:
        df = pd.read_excel(file)
        # Nettoyage des colonnes
        df.columns = df.columns.str.strip().str.upper()
        return df
    except Exception:
        # Si Excel échoue, on ne s'arrête pas, on passe au plan B.
        pass

    # 2. Tentative : Lire comme un CSV avec Point-Virgule (Format France/Excel)
    try:
        file.seek(0) # Rembobiner le fichier pour le relire du début
        df = pd.read_csv(file, sep=';', encoding='latin-1', on_bad_lines='skip')
        if len(df.columns) > 1: # Si on a bien séparé les colonnes
            df.columns = df.columns.str.strip().str.upper()
            return df
    except Exception:
        pass

    # 3. Tentative : Lire comme un CSV avec Virgule (Format Standard)
    try:
        file.seek(0)
        df = pd.read_csv(file, sep=',', encoding='utf-8', on_bad_lines='skip')
        if len(df.columns) > 1:
            df.columns = df.columns.str.strip().str.upper()
            return df
    except Exception:
        pass

    # 4. Si tout échoue
    return None

def extract_year(filename):
    match = re.search(r'20\d{2}', filename)
    return int(match.group(0)) if match else None

# --- SIDEBAR : IMPORTATION ---
st.sidebar.header("📂 1. Données Sources")
uploaded_files = st.sidebar.file_uploader("Chargez vos fichiers (Excel ou CSV)", accept_multiple_files=True)

data_dict = {}
combined_df = pd.DataFrame()

if uploaded_files:
    temp_dfs = []
    for f in uploaded_files:
        year = extract_year(f.name)
        if year:
            # On utilise la nouvelle fonction de chargement robuste
            df = load_data(f)
            
            if df is not None:
                # Renommage intelligent des colonnes pour standardiser (CDC / ARACT)
                rename_map = {
                    'SERVICE / SECTEUR': 'SERVICE', 'SECTEUR': 'SERVICE', 'UNITÉ': 'SERVICE',
                    'AGENCE': 'SERVICE', 'ENTITÉ': 'SERVICE',
                    'DURÉE DU TRAVAIL': 'TEMPS_TRAVAIL',
                    'ARRIVÉE': 'ENTREE', 'DATE ENTREE': 'ENTREE',
                    'DATE NAISSANCE': 'NAISSANCE'
                }
                # On applique le renommage si la colonne existe
                new_cols = {k: v for k, v in rename_map.items() if k in df.columns}
                df = df.rename(columns=new_cols)
                
                df['ANNEE_FICH'] = year
                data_dict[str(year)] = df
                temp_dfs.append(df)
            else:
                st.sidebar.error(f"❌ Impossible de lire : {f.name}")
        else:
            st.sidebar.warning(f"⚠️ Pas d'année trouvée dans le nom : {f.name}")
    
    if temp_dfs:
        combined_df = pd.concat(temp_dfs, ignore_index=True)
        sorted_years = sorted(data_dict.keys())
        st.sidebar.success(f"✅ {len(data_dict)} années chargées : {min(sorted_years)} - {max(sorted_years)}")
        
        # --- PRÉ-CALCULS ---
        cols = combined_df.columns
        # Recherche souple des colonnes
        col_naiss = next((c for c in cols if 'NAISS' in c), None)
        col_entree = next((c for c in cols if 'ENTREE' in c or 'ARRIVEE' in c), None)
        col_service = next((c for c in cols if 'SERVICE' in c or 'AGENCE' in c), 'SERVICE')
        col_emploi = next((c for c in cols if 'EMPLOI' in c or 'POSTE' in c or 'FONCTION' in c), 'EMPLOI')

        if col_naiss and col_entree:
             combined_df['Date_Naiss'] = pd.to_datetime(combined_df[col_naiss], dayfirst=True, errors='coerce')
             combined_df['Date_Entree'] = pd.to_datetime(combined_df[col_entree], dayfirst=True, errors='coerce')
             
             combined_df['AGE_CALC'] = combined_df['ANNEE_FICH'] - combined_df['Date_Naiss'].dt.year
             combined_df['ANC_CALC'] = combined_df['ANNEE_FICH'] - combined_df['Date_Entree'].dt.year
             
             # Filtre données aberrantes
             combined_df = combined_df[(combined_df['AGE_CALC'] > 15) & (combined_df['AGE_CALC'] < 80)]
        else:
            st.error(f"⚠️ Colonnes manquantes. Trouvé : Naissance={col_naiss}, Entrée={col_entree}")

# --- ANALYSE ---
if not combined_df.empty:
    tabs = st.tabs(["📉 Flux (Histo Décalé)", "📍 Structure (Nuage)", "📊 Absentéisme (Barres)"])

    # --- TAB 1 : FLUX ---
    with tabs[0]:
        st.header("Analyse des Entrées/Sorties")
        col_var = st.radio("Analyser :", ["Âge", "Ancienneté"], horizontal=True)
        c1, c2 = st.columns(2)
        y_start = c1.selectbox("Départ (Passé)", sorted_years, index=0)
        y_end = c2.selectbox("Arrivée (Réel)", sorted_years, index=len(sorted_years)-1)
        
        if y_start and y_end:
            shift = int(y_end) - int(y_start)
            df_p = data_dict[y_start].copy()
            df_r = data_dict[y_end].copy()
            
            if col_var == "Âge":
                df_p['VAL'] = (int(y_start) - pd.to_datetime(df_p[col_naiss], dayfirst=True, errors='coerce').dt.year) + shift
                df_r['VAL'] = int(y_end) - pd.to_datetime(df_r[col_naiss], dayfirst=True, errors='coerce').dt.year
            else:
                df_p['VAL'] = (int(y_start) - pd.to_datetime(df_p[col_entree], dayfirst=True, errors='coerce').dt.year) + shift
                df_r['VAL'] = int(y_end) - pd.to_datetime(df_r[col_entree], dayfirst=True, errors='coerce').dt.year
            
            idx = sorted([x for x in (set(df_p['VAL'].dropna()) | set(df_r['VAL'].dropna())) if 0 <= x <= 70])
            vc_p = df_p['VAL'].value_counts().reindex(idx, fill_value=0)
            vc_r = df_r['VAL'].value_counts().reindex(idx, fill_value=0)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=vc_p.index, y=vc_p.values, name=f"Théorique ({y_start}+{shift})", line=dict(color='orange', width=2, dash='dash')))
            fig.add_trace(go.Scatter(x=vc_r.index, y=vc_r.values, name=f"Réel ({y_end})", fill='tozeroy', line=dict(color='#1f77b4', width=3)))
            st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2 : STRUCTURE ---
    with tabs[1]:
        st.header("Cartographie")
        y_tri = st.selectbox("Année", sorted_years, index=len(sorted_years)-1, key='sc_y')
        grp = st.selectbox("Groupe", [c for c in [col_service, col_emploi, 'SEXE'] if c in combined_df.columns], key='sc_g')
        
        df_t = data_dict[y_tri].copy()
        df_t['VAL_AGE'] = int(y_tri) - pd.to_datetime(df_t[col_naiss], dayfirst=True, errors='coerce').dt.year
        
        stats = []
        for n, g in df_t.groupby(grp):
            if len(g) > 2:
                stats.append({
                    'Groupe': n, 'Eff': len(g),
                    'Jeunes': (len(g[g['VAL_AGE']<30])/len(g))*100,
                    'Seniors': (len(g[g['VAL_AGE']>=50])/len(g))*100
                })
        
        if stats:
            fig = px.scatter(pd.DataFrame(stats), x='Seniors', y='Jeunes', size='Eff', color='Groupe', text='Groupe', 
                             title="Structure Démographique", labels={'Seniors': '% >50 ans', 'Jeunes': '% <30 ans'})
            fig.update_layout(xaxis=dict(range=[0,100]), yaxis=dict(range=[0,100]))
            st.plotly_chart(fig, use_container_width=True)

    # --- TAB 3 : ABSENTÉISME ---
    with tabs[2]:
        st.header("Comparatif Absentéisme (Poids)")
        col_f, col_v = st.columns([1,3])
        with col_f:
            ys = st.multiselect("Années", sorted_years, default=sorted_years)
            ga = st.selectbox("Axe", [c for c in [col_service, col_emploi, 'SEXE'] if c in combined_df.columns], key='ab_g')
            # Détection colonnes abs
            abs_cols = [c for c in combined_df.columns if 'NB J' in c or 'NB H' in c]
            ms = st.multiselect("Indicateurs", abs_cols, default=abs_cols[:2] if abs_cols else None)
        
        if ys and ga and ms:
            df_c = combined_df[combined_df['ANNEE_FICH'].isin([int(y) for y in ys])]
            
            data = []
            # 1. Effectif
            tmp = df_c.groupby(ga).size().reset_index(name='Valeur')
            tmp['Type'] = "1. Poids Effectif"
            data.append(tmp)
            
            # 2. Absences
            for m in ms:
                tmp = df_c.groupby(ga)[m].sum().reset_index()
                tmp.columns = [ga, 'Valeur']
                tmp['Type'] = f"2. {m}"
                data.append(tmp)
            
            df_plot = pd.concat(data)
            
            with col_v:
                fig = px.bar(df_plot, x='Type', y='Valeur', color=ga, title="Répartition Normalisée (100%)", text_auto='.1f')
                fig.update_layout(barmode='stack', barnorm='percent', yaxis_title="% Part")
                fig.update_traces(hovertemplate='%{y:.1f}%<br>%{x}')
                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Chargez vos fichiers (CSV ou Excel).")