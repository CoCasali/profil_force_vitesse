import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import signal
from numpy import linalg as LA
import os
from sklearn.linear_model import LinearRegression


# /Users/corentincasali/SELFIT/CODE/PFV/ANALYSE/01_DATA

# Full size page
st.set_page_config(
    page_title = "ANALYSE - PROFIL FORCE-VITESSE",
    page_icon = "🔬",
    layout="wide")

# Hide Hamburger (Top Right Corner) and "Made with Streamlit" footer:
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

pd.options.display.float_format = '{:,.3f}'.format


# Function
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

def convert_df(input_df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return input_df.to_html(escape=False, index = False, justify = "center")

############################################################
# CONSTANTE 
############################################################
FC = 20
Fr = 1000
FreqAcquis = 1 / Fr
Inertie = 13.8
Resistance = 0.0001
Pignons = 14
Volant = 0.2571
Plateau = 52
Fs = 1000.0      # sample rate, Hz
cutoff = 20      # desired cutoff frequency of the filter, Hz
order = 4
CoefVitAngulairePedalier = Pignons/(Volant*Plateau)
CritereRecherche = 0.95
indice_delta = 45
nyq = 0.5 * Fs      # Frequence echantillonnage / 2

############################################################
# FONCTIONS
############################################################
# Fonction du filtre butterworth low_pass
def lecture_fichier(files):
    """
    files : path+files you want to analyse
    """
    data = pd.read_excel(files,sheet_name = "DATA_CALC",header=1)
    data2 = pd.read_excel(files,sheet_name = "DEMI_CYCLES")

    # Filter requirements.
    T = len(data)/1000       # Sample Period
    nyq = 0.5 * Fs      # Frequence echantillonnage / 2
    n = int(T * Fs)

    return(data,data2,nyq)

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data, padlen=len(data)-1)
    return y

def calcul_dataCalc(data):
    # Récupération des valeurs du fichier
    temps = data["Temps"]
    force = data["FAcquis(N)"]
    deplacement = data["Depl Acquis(m).1"]
    vitesse = data["Vit(m.s-1).1"]
    acc = data["Acc(m.s-2).1"]
    ForceInertie = acc * Inertie + Resistance
    ForceTotale = force + ForceInertie
    Puissance = ForceTotale * vitesse 
    VitesseAngulaire = vitesse * CoefVitAngulairePedalier

    dataCalc = pd.DataFrame(np.transpose([temps,deplacement,vitesse,acc,ForceTotale,Puissance,VitesseAngulaire]))
    dataCalc.rename(columns={
        0:"Temps",
        1:"Déplacement",
        2:"Vitesse",
        3:"Accélération",
        4:"Force",
        5:"Puissance",
        6:"VitesseAngulaire"}, inplace = True)

    return(dataCalc,Puissance)

def detection_point(dataCalc,Puissance):
    # Application du filtre : 
    puissance_filt = butter_lowpass_filter(Puissance, cutoff, Fs, order)

    # Détection des points en moyenne
    x = puissance_filt
    peaks, _ = signal.find_peaks(x, distance = 100,prominence=100)
    vallees, _ = signal.find_peaks(-x, distance = 100,prominence=100, height=-x.mean()-200)

    indexVal = []
    for i in range (0,len(vallees)):
        indexVal.append(dataCalc.iloc[vallees[i]-indice_delta:vallees[i]+indice_delta]["Puissance"].idxmin())

    indexPic = []
    for i in range (0,len(peaks)):
        indexPic.append(dataCalc.iloc[peaks[i]-indice_delta:peaks[i]+indice_delta]["Puissance"].idxmax())

    df_indexVal = pd.DataFrame(indexVal, columns = ['index'])
    df_indexVal['origineV'] = 'vallee'
    df_indexVal.set_index('index',inplace=True)
    df_indexPic = pd.DataFrame(indexPic, columns = ['index'])
    df_indexPic['origineP'] = 'pic'
    df_indexPic.set_index('index',inplace=True)

    resultat_index = pd.concat([df_indexPic,df_indexVal], axis=1)
    resultat_index['origine_final'] = resultat_index['origineP'].fillna(resultat_index['origineV'])
    resultat_index.reset_index(inplace=True)
    resultat_index.drop(["origineP","origineV"], axis=1, inplace=True)

            # Reset index dataCalc
    dataCalc_index_reset = dataCalc.reset_index()
    resultat_index = pd.merge(resultat_index,dataCalc_index_reset, on='index', how='inner')
    resultat_index = resultat_index.sort_values("index").reset_index(drop=True)                 # changement, on sort les values par index, et on reset les index pour avoir quelques choses de propre

    for i in range(0,len(resultat_index)-1):
        # Détection des doubles pics
        if ((resultat_index.iloc[i]['origine_final'] == 'pic') and (resultat_index.iloc[i+1]['origine_final'] == 'pic')):
            resultat_index.at[i,'doublepic'] = 1
        else:
            resultat_index.at[i,'doublepic'] = 0
        # Détection des doubles vallées
        if ((resultat_index.iloc[i]['origine_final'] == 'vallee') and (resultat_index.iloc[i+1]['origine_final'] == 'vallee')):
            resultat_index.at[i,'doubleval'] = 1
        else:
            resultat_index.at[i,'doubleval'] = 0

    resultat_index.fillna(0, inplace=True)

    # Erreur dans le boucle, le .at(i+1) ne fonctionne pas

    for i in range(0,len(resultat_index)-1):
        # choix du double pic à supprimer
        if resultat_index.iloc[i]['doublepic'] == 1:
            if resultat_index.iloc[i]['Puissance'] < resultat_index.iloc[i+1]['Puissance']:
                resultat_index.at[i,'doublepic_2'] = 1
                resultat_index.at[i+1,'doublepic_2'] = 0
            elif resultat_index.iloc[i]['Puissance'] > resultat_index.iloc[i+1]['Puissance']:
                resultat_index.at[i,'doublepic_2'] = 0
                resultat_index.at[i+1,'doublepic_2'] = 1
        else:
            resultat_index.at[i+1,'doublepic_2'] = 0 

        # # choix du double vallee à supprimer
        if resultat_index.iloc[i]['doubleval'] == 1:
            if resultat_index.iloc[i]['Puissance'] < resultat_index.iloc[i+1]['Puissance']:
                resultat_index.at[i,'doubleval_2'] = 0
                resultat_index.at[i+1,'doubleval_2'] = 1
            elif resultat_index.iloc[i]['Puissance'] > resultat_index.iloc[i+1]['Puissance']:
                resultat_index.at[i,'doubleval_2'] = 1
                resultat_index.at[i+1,'doubleval_2'] = 0
        else:
            resultat_index.at[i+1,'doubleval_2'] = 0 

    resultat_index.fillna(0, inplace=True)

    resultat_index.drop(resultat_index[resultat_index["doublepic_2"]==1].index, inplace=True,errors='ignore')
    resultat_index.drop(resultat_index[resultat_index["doubleval_2"]==1].index, inplace=True,errors='ignore')
    # Attribution des données peak et val du filtre dans des dataframes
    dataPeak = dataCalc.iloc[indexPic].copy()
    dataVal = dataCalc.iloc[indexVal].copy()

    # Récupération des vrai valeurs de puissance et d'index avec
    # Premier pic ou premiere vallées > 20 Watts

    # Suppression des valeurs ne dépassant pas 20 Watts et inférieur à 2 secondes
    dataVal.drop(index=dataVal.loc[(dataVal["Puissance"]<20)&(dataVal["Temps"]<2)].index.values,inplace=True)

    return(resultat_index,dataPeak,dataVal)

def demi_cycle(dataCalc, resultat_index):
    # Ajout de la première ligne
    # dataVal = pd.concat([pd.DataFrame(dataCalc.iloc[0]).transpose(),dataVal], ignore_index = False, axis = 0)
    # newIndexVal = dataVal.index.values
    newIndexVal = np.array(resultat_index.loc[resultat_index["origine_final"]=="vallee"]['index'])
    newIndexVal = np.insert(newIndexVal,0,0)
    newIndexPic = np.array(resultat_index.loc[resultat_index["origine_final"]=="pic"]['index'])

    # ## Calcul des DEMI_CYCLES
    # Calcul du temps, déplacement, vitesse, acceleration, force, puissance et vitesse angulaire
    DC_temps = []
    for i in range (0,len(newIndexVal)-1):
        DC_temps.append((dataCalc["Temps"].iloc[newIndexVal[i+1]]+dataCalc["Temps"].iloc[newIndexVal[i]])/2)

    DC_puissance = []
    for i in range (0,len(newIndexVal)-1):
        DC_puissance.append(np.mean(dataCalc["Puissance"].iloc[newIndexVal[i]:newIndexVal[i+1]]))

    DC_vitesse = []
    for i in range (0,len(newIndexVal)-1):
        DC_vitesse.append(np.mean(dataCalc["Vitesse"].iloc[newIndexVal[i]:newIndexVal[i+1]]))

    DC_force = []
    for i in range (0,len(newIndexVal)-1):
        DC_force.append(np.mean(dataCalc["Force"].iloc[newIndexVal[i]:newIndexVal[i+1]]))

    DC_vitAng = []
    for i in range (0,len(newIndexVal)-1):
        DC_vitAng.append(np.mean(dataCalc["VitesseAngulaire"].iloc[newIndexVal[i]:newIndexVal[i+1]]))

    ## DATAFRAME DEMI_CYCLE
    DEMI_CYCLE = pd.DataFrame(np.transpose([DC_temps,DC_vitesse,DC_force, DC_puissance,DC_vitAng]))
    DEMI_CYCLE.rename(columns={
        0:"Temps",
        1:"Vitesse",
        2:"Force",
        3:"Puissance",
        4:"VitAng"}, inplace = True)

    return(DEMI_CYCLE)

def nbvaleurmax(DEMI_CYCLE):
    # --------2 EME MÉTHODE--------
    # Calcul de Vmax (on ne compte pas le dernier cycle)
    Vmax = DEMI_CYCLE["Vitesse"].max()
    # Fixation du critère de recherche à 0.99
    nbValeurMax = DEMI_CYCLE[DEMI_CYCLE["Vitesse"]>=CritereRecherche*Vmax].index[0]
    # print("Critère de recherche de :", round(CritereRecherche*Vmax,3))
    # print("Nombre de valeurs choisies :",nbValeurMax)

    return(nbValeurMax)

def parametre_0(DEMI_CYCLE,nbValeurMax,data2,files):
    # Calcul de régression
    x = DEMI_CYCLE["VitAng"][2:nbValeurMax].values.reshape(-1,1)
    y = DEMI_CYCLE["Force"][2:nbValeurMax].values.reshape(-1,1)
    reg = LinearRegression().fit(x,y)
    B0 = reg.intercept_[0]
    B1 = reg.coef_.flatten()[0]    # double array transform in 1D array
    R2 = reg.score(x,y)

    # Calcul des différents paramètres
    F0 = B0
    V0 = -B0/B1
    Pmax = V0*F0/4

    # Récupération des données de Rodolphe
    aRodolphe = data2.iloc[1,9]
    bRodolphe = data2.iloc[1,10]
    pmaxRodolphe = data2.iloc[1,15]
    R2_Rodolphe = round(data2.iloc[1,11],5)

    # Dictionary 
    cache = {
        "Nom" : files[:-6],
        "Essai" : files[-5],
        "A" : B1,
        "B" : B0,
        "R2" : R2,
        "x" : x,
        "y" : y
    }
    df = pd.DataFrame([cache])
    foo = df[df.columns].astype(str)
    return(foo)

def parametre_1(DEMI_CYCLE,df,nbValeurMax,data2,files):
    # Calcul de régression
    x = DEMI_CYCLE["VitAng"][2:nbValeurMax].values.reshape(-1,1)
    y = DEMI_CYCLE["Force"][2:nbValeurMax].values.reshape(-1,1)
    reg = LinearRegression().fit(x,y)
    B0 = reg.intercept_[0]
    B1 = reg.coef_.flatten()[0]    # double array transform in 1D array
    R2 = reg.score(x,y)

    # Calcul des différents paramètres
    F0 = B0
    V0 = -B0/B1
    Pmax = V0*F0/4
    
    # Récupération des données de Rodolphe
    aRodolphe = data2.iloc[1,9]
    bRodolphe = data2.iloc[1,10]
    pmaxRodolphe = data2.iloc[1,15]
    R2_Rodolphe = round(data2.iloc[1,11],5)

    # Dictionary 
    cache = {
        "Nom" : files[:-6],
        "Essai" : files[-5],
        "A" : B1,
        "B" : B0,
        "R2" : R2,
        "x" : x,
        "y" : y
    }
    df = pd.concat([df,pd.DataFrame([cache])],ignore_index=True)
    return(df)

def create_third(data1,data2):
    # DF for the merge
    data = pd.concat(data1,data2)
    cache1 = data.loc[data["Essai"]=='1'].reset_index(drop=True)
    cache2 = data.loc[data["Essai"]=='2'].reset_index(drop=True)

    data = pd.merge(cache1, cache2,
            on = "Nom",
            how = "inner",
            suffixes=('_1','_2')
    )

    # Drop columns 
    data.drop(["Essai_1","Essai_2"],axis = 1, inplace = True)

    # Calcul des paramètres sur le mix des deux premiers essais
    cache_x = []
    cache_y = []
    for i in range(len(data)):
        cache_x.append(np.concatenate((data["x_1"][i],data["x_2"][i]),axis=0))
        cache_y.append(np.concatenate((data["y_1"][i],data["y_2"][i]),axis=0))

    data["x_3"] = cache_x
    data["y_3"] = cache_y

    cache_R2 = []
    cache_A = []
    cache_B = []
    for i in range(len(data)):
        x = data["x_3"][i]
        y = data["y_3"][i]
        reg = LinearRegression().fit(x,y)
        cache_R2.append(
            reg.score(x,y)
        )
        cache_A.append(
            reg.coef_.flatten()[0]
        )
        cache_B.append(
            reg.intercept_[0]
        )

    data["A_3"] = cache_A
    data["B_3"] = cache_B
    data["R2_3"] = cache_R2

    # Drop columns
    data.drop(["x_1","x_2","x_3","y_1","y_2","y_3"],axis = 1, inplace = True)
    return(data)

def calcul_pmax(a,b):
    F0 = b
    V0 = -b/a 
    Pmax = V0*F0/4
    return(Pmax)

def parametre_PFV(data, delta):
    x = data["VitAng"][2:nbValeurMax+delta].values.reshape(-1,1)
    y = data["Force"][2:nbValeurMax+delta].values.reshape(-1,1)
    reg = LinearRegression().fit(x,y)
    B0 = reg.intercept_[0]
    B1 = reg.coef_.flatten()[0]
    R2 = reg.score(x,y)

    return(B0, B1, R2, x, y)
    
def df_R2_styling():
    # 5 points à + et -
    point_r2 = 5
    dimension_r2 = np.arange(-point_r2,point_r2+1)

    resultat = np.array([])
    a = np.array([])
    b = np.array([])
    Pmax = np.array([])
    columns_name = []

    for i in dimension_r2:
        B0, B1, R2, x, y = parametre_PFV(DEMI_CYCLE,i)
        a = np.append(a,B1)
        b = np.append(b,B0)
        Pmax = np.append(Pmax,round((-B0/B1)*B0/4,3))
        resultat = np.append(resultat,R2)
        columns_name = np.append(columns_name, int(i))

    index = ["a","b","Pmax","r2"]
    df_R2 = pd.DataFrame(np.vstack((a,b,Pmax,resultat)), columns = columns_name, index = index)
    df_R2_styling = df_R2.style.applymap(lambda x: 'background-color : #BEFD50' if (x > 0.90 and x < 1) else ('background-color : #EE827F' if (x < 0.90 and x > 0) else ''))
    return(df_R2_styling,df_R2, point_r2)
#############################################
st.markdown(f'<h1 style="background-color:#6497b1;border-radius:14px;text-align:center;">{"ANALYSE : PROFIL FORCE-VITESSE"}</h1>', unsafe_allow_html=True)
f'*version 2.00 - par Corentin C.*'
f"""Dashboard : permet d'analyser les sprints Wingate d'un sujet de créer automatiquement son 3ème sprint. Il est possible de sélectionner le nombre de points pour chaque sprint, afin d'ajuster le R2."""

st.markdown(f'<h2 style="background-color:#b1cbd8;border-radius:14px;text-align:center;">{"LECTURE DES FICHIERS"}</h2>', unsafe_allow_html=True)
f"Pour importer un fichier .xls, il faut d'abord insérez le chemin du dossier comportant les dossiers PRE | POST des sprints. Par la suite, vous pourrez sélectionner un sprint."
st.info("""Pour l'organisation de vos dossiers, merci de suivre l'exemple ci-dessous : \n
Dans la sélection du fichier sprint : ne sélectionner que le fichier avec l'indice _1. Automatiquement, le script ira chercher le fichier du deuxième esssai (i.e. : xxxx_2). 
""", icon="ℹ️")

code = """
CHEMIN DE VOS DOSSIERS              <-- à insérer dans la première case en dessous
├── PRE                             <-- sépération en deux (si des essais PRE | POST), sinon plusieurs sépérations possibles
└── POST
    ├── NOM_1.xls                   <-- Exemple d'essai (ici un sujet avec son premier essai)
    ├── NOM_2.xls                    
    ├── AUTRENOM_1.xls              <-- Nouveau sujet avec un nouvel essai
    └── AUTRENOM_2.xls              
"""

st.code(code)

col1, col2 = st.columns(2)
with col1:
    filename = st.file_uploader("Fichier n°1")

with col2: 
    filename_2 = st.file_uploader("Fichier n°2")

if filename:
    col1, col2 = st.columns(2)
    with col1:
        data, data2, nyq = lecture_fichier(filename)
        dataCalc, Puissance = calcul_dataCalc(data)
        resultat_index,dataPeak,dataVal = detection_point(dataCalc,Puissance)
        DEMI_CYCLE = demi_cycle(dataCalc,resultat_index)
        nbValeurMax = nbvaleurmax(DEMI_CYCLE)

        df_R2_style,df_R2, point_r2 = df_R2_styling()
        f"*Tableau pour le fichier 1*"
        st.write(df_R2_style)

        newIndexVal = np.array(resultat_index.loc[resultat_index["origine_final"]=="vallee"]['index'])
        newIndexVal = np.insert(newIndexVal,0,0)
        newIndexPic = np.array(resultat_index.loc[resultat_index["origine_final"]=="pic"]['index'])

        f"### Graphique Puissance"
        nombre_r2 = st.number_input("Sélection du nombre de point en plus ou en moins : ", min_value=None, max_value=None, value=0, step=1,key=0)

        # Graphique Puissance
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=Puissance,
            line_color = "#99aab5",
            name = "Puissance"
        ))

        # Marqueurs Pic
        fig.add_trace(go.Scatter(
            x = newIndexPic,
            y = Puissance[newIndexPic],
            mode = "markers",
            marker = dict(
                size = 8,
                symbol = 'x-thin',
                line=dict(
                    color = "Blue",
                    width = 1
            )), name = "Pic"
        ))

        # Marqueur pic OK
        fig.add_trace(go.Scatter(
            x = newIndexPic[2:nbValeurMax+nombre_r2],
            y = Puissance[newIndexPic][2:nbValeurMax+nombre_r2],
            mode = "markers",
            marker = dict(
                size = 8,
                symbol = 'x-thin',
                line=dict(
                    color = "Green",
                    width = 2
            )), name = "Pic - OK"
        ))

        # Marqueur vallée
        fig.add_trace(go.Scatter(
            x = newIndexVal,
            y = Puissance[newIndexVal],
            mode = "markers",
            marker = dict(
                size = 8,
                symbol = 'x-thin',
                line=dict(
                    color = "Red",
                    width = 1
            )), name = "Vallée"
        ))

        fig.update_layout(
            xaxis_title="Temps",
            yaxis_title="Puissance (W)",
            font=dict(
                family="Verdana, monospace",
                size=12,
                color="Black"
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        f"*Paramètre du profil force-vitesse (1)*"
        a =  round(df_R2.iloc[0,point_r2],3)
        b =  round(df_R2.iloc[1,point_r2],3)
        Pmax =  round(df_R2.iloc[2,point_r2],3)
        r2 = round(df_R2.iloc[3,point_r2],3)

        cache = {
            "Type":"Basique",
            "A":a,
            "B":b,
            "Pmax": Pmax,
            "R2" : r2
        }

        B0, B1, R2, x, y = parametre_PFV(DEMI_CYCLE,nombre_r2) 
        # Calcul des différents paramètres
        a_mod_1 = round(B1,3)
        b_mod_1 = round(B0,3)
        Pmax_mod_1 = round((-B0/B1)*B0/4,3)
        r2_mod_1 = round(R2,3)


        cache_mod = {
            "Type":"Modifié",
            "A":a_mod_1,
            "B":b_mod_1,
            "Pmax": Pmax_mod_1,
            "R2" : r2_mod_1
        }

        cache_delta = {
            "Type":"Delta",
            "A":a-a_mod_1,
            "B":b_mod_1-b,
            "Pmax": Pmax_mod_1-Pmax,
            "R2" : r2_mod_1-r2
        }

        cache_delta_percentage = {
            "Type":"%",
            "A":(a-a_mod_1)/a * 100,
            "B":(b_mod_1-b)/b* 100,
            "Pmax": (Pmax_mod_1-Pmax)/Pmax* 100,
            "R2" : (r2_mod_1-r2)/r2* 100
        }

        cache_df = pd.DataFrame([cache,cache_mod,cache_delta,cache_delta_percentage])
        html = convert_df(cache_df)
        st.markdown(
            html, 
            unsafe_allow_html=True
        )

        # Récupération x et y
        x_1_0 = DEMI_CYCLE["VitAng"][2:nbValeurMax]
        y_1_0 = DEMI_CYCLE["Force"][2:nbValeurMax]
        x_1 = DEMI_CYCLE["VitAng"][2:nbValeurMax+nombre_r2]
        y_1 = DEMI_CYCLE["Force"][2:nbValeurMax+nombre_r2]

        # Relation force-Vitesse
        f"*Relation force-vitesse (1)*"
        fig = px.scatter(
            x = x_1,
            y = y_1,
            trendline = 'ols',trendline_color_override="lightgray")


        fig.update_yaxes(rangemode="tozero")

        fig.update_traces(
            marker=dict(
                size=10,
                color = "Red"
            )
        )

        fig.update_layout(
            xaxis_title="Vitesse (rad.s-1)",
            yaxis_title="Force (N)",
            font=dict(
                family="Verdana",
                size=12,
                color="Black"
            )
        )

        st.plotly_chart(fig, use_container_width=False)


    with col2:
        data, data2, nyq = lecture_fichier(filename_2)
        dataCalc, Puissance = calcul_dataCalc(data)
        resultat_index,dataPeak,dataVal = detection_point(dataCalc,Puissance)
        DEMI_CYCLE = demi_cycle(dataCalc,resultat_index)
        nbValeurMax = nbvaleurmax(DEMI_CYCLE)

        df_R2_style,df_R2, point_r2 = df_R2_styling()
        f"*Tableau pour le fichier 2*"
        st.write(df_R2_style)

        newIndexVal = np.array(resultat_index.loc[resultat_index["origine_final"]=="vallee"]['index'])
        newIndexVal = np.insert(newIndexVal,0,0)
        newIndexPic = np.array(resultat_index.loc[resultat_index["origine_final"]=="pic"]['index'])

        f"### Graphique Puissance"
        nombre_r2_2 = st.number_input("Sélection du nombre de point en plus ou en moins : ", min_value=None, max_value=None, value=0, step=1,key=1)

        # Graphique Puissance
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=Puissance,
            line_color = "#99aab5",
            name = "Puissance"
        ))

        # Marqueurs Pic
        fig.add_trace(go.Scatter(
            x = newIndexPic,
            y = Puissance[newIndexPic],
            mode = "markers",
            marker = dict(
                size = 8,
                symbol = 'x-thin',
                line=dict(
                    color = "Blue",
                    width = 1
            )), name = "Pic"
        ))

        # Marqueur pic OK
        fig.add_trace(go.Scatter(
            x = newIndexPic[2:nbValeurMax+nombre_r2_2],
            y = Puissance[newIndexPic][2:nbValeurMax+nombre_r2_2],
            mode = "markers",
            marker = dict(
                size = 8,
                symbol = 'x-thin',
                line=dict(
                    color = "Green",
                    width = 2
            )), name = "Pic - OK"
        ))

        # Marqueur vallée
        fig.add_trace(go.Scatter(
            x = newIndexVal,
            y = Puissance[newIndexVal],
            mode = "markers",
            marker = dict(
                size = 8,
                symbol = 'x-thin',
                line=dict(
                    color = "Red",
                    width = 1
            )), name = "Vallée"
        ))

        fig.update_layout(
            xaxis_title="Temps",
            yaxis_title="Puissance (W)",
            font=dict(
                family="Verdana, monospace",
                size=12,
                color="Black"
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        f"*Paramètre du profil force-vitesse (2)*"
        a =  round(df_R2.iloc[0,point_r2],3)
        b =  round(df_R2.iloc[1,point_r2],3)
        Pmax =  round(df_R2.iloc[2,point_r2],3)
        r2 = round(df_R2.iloc[3,point_r2],3)

        cache = {
            "Type":"Basique",
            "A":a,
            "B":b,
            "Pmax": Pmax,
            "R2" : r2
        }

        B0, B1, R2, x, y = parametre_PFV(DEMI_CYCLE,nombre_r2_2) 
        # Calcul des différents paramètres
        a_mod_2 = round(B1,3)
        b_mod_2 = round(B0,3)
        Pmax_mod_2 = round((-B0/B1)*B0/4,3)
        r2_mod_2 = round(R2,3)


        cache_mod = {
            "Type":"Modifié",
            "A":a_mod_2,
            "B":b_mod_2,
            "Pmax": Pmax_mod_2,
            "R2" : r2_mod_2
        }

        cache_delta = {
            "Type":"Delta",
            "A":a-a_mod_2,
            "B":b_mod_2-b,
            "Pmax": Pmax_mod_2-Pmax,
            "R2" : r2_mod_2-r2
        }

        cache_delta_percentage = {
            "Type":"%",
            "A":(a-a_mod_2)/a * 100,
            "B":(b_mod_2-b)/b* 100,
            "Pmax": (Pmax_mod_2-Pmax)/Pmax* 100,
            "R2" : (r2_mod_2-r2)/r2* 100
        }

        cache_df = pd.DataFrame([cache,cache_mod,cache_delta,cache_delta_percentage])
        html = convert_df(cache_df)
        st.markdown(
            html, 
            unsafe_allow_html=True
        )

        # Récupération de x et y
        x_2_0 = DEMI_CYCLE["VitAng"][2:nbValeurMax]
        y_2_0 = DEMI_CYCLE["Force"][2:nbValeurMax]
        x_2 = DEMI_CYCLE["VitAng"][2:nbValeurMax+nombre_r2_2]
        y_2 = DEMI_CYCLE["Force"][2:nbValeurMax+nombre_r2_2]

        # Relation force-Vitesse
        f"*Relation force-vitesse (2)*"
        fig = px.scatter(
            x = x_2,
            y = y_2,
            trendline = 'ols',trendline_color_override="lightgray")


        fig.update_yaxes(rangemode="tozero")

        fig.update_traces(
            marker=dict(
                size=10,
                color = "Blue"
            )
        )

        fig.update_layout(
            xaxis_title="Vitesse (rad.s-1)",
            yaxis_title="Force (N)",
            font=dict(
                family="Verdana",
                size=12,
                color="Black"
            )
        )

        st.plotly_chart(fig, use_container_width=False)

    st.markdown(f'<h2 style="background-color:#b1cbd8;border-radius:14px;text-align:center;">{"CREATION DU 3ÈME ESSAI"}</h2>', unsafe_allow_html=True)
    cache1 = pd.DataFrame([x_1_0,y_1_0]).transpose()
    cache2 = pd.DataFrame([x_2_0,y_2_0]).transpose()
    parametre_df_essai3_0 = pd.concat([cache1,cache2]).reset_index(drop=True)


    cache1 = pd.DataFrame([x_1,y_1]).transpose()
    cache1["essai"] = 1

    cache2 = pd.DataFrame([x_2,y_2]).transpose()
    cache2["essai"] = 2

    df_essai_3 = pd.concat([cache1,cache2]).reset_index(drop=True)
    df_essai_3["essai"] = df_essai_3["essai"].astype("str")


    # Tableau paramètres
    f"Paramètre du profil force-vitesse (3)"
    x = parametre_df_essai3_0["VitAng"].values.reshape(-1,1)
    y = parametre_df_essai3_0["Force"].values.reshape(-1,1)
    reg = LinearRegression().fit(x,y)
    B0 = reg.intercept_[0]
    B1 = reg.coef_.flatten()[0]
    R2 = reg.score(x,y)

    a = round(B1,3)
    b = round(B0,3)
    Pmax = round((-B0/B1)*B0/4,3)
    r2 = round(R2,3)

    cache = {
        "Type":"Basique",
        "A":a,
        "B":b,
        "Pmax": Pmax,
        "R2" : r2
    }

    x = df_essai_3["VitAng"].values.reshape(-1,1)
    y = df_essai_3["Force"].values.reshape(-1,1)

    reg = LinearRegression().fit(x,y)
    B0 = reg.intercept_[0]
    B1 = reg.coef_.flatten()[0]
    R2 = reg.score(x,y)

    a_mod_3 = round(B1,3)
    b_mod_3 = round(B0,3)
    Pmax_mod_3 = round((-B0/B1)*B0/4,3)
    r2_mod_3 = round(R2,3)


    cache_mod = {
        "Type":"Modifié",
        "A":a_mod_3,
        "B":b_mod_3,
        "Pmax": Pmax_mod_3,
        "R2" : r2_mod_3
    }

    cache_delta = {
        "Type":"Delta",
        "A":a-a_mod_3,
        "B":b_mod_3-b,
        "Pmax": Pmax_mod_3-Pmax,
        "R2" : r2_mod_3-r2
    }

    cache_delta_percentage = {
        "Type":"%",
        "A":(a-a_mod_3)/a * 100,
        "B":(b_mod_3-b)/b* 100,
        "Pmax": (Pmax_mod_3-Pmax)/Pmax* 100,
        "R2" : (r2_mod_3-r2)/r2* 100
    }

    cache_df = pd.DataFrame([cache,cache_mod,cache_delta,cache_delta_percentage])
    html = convert_df(cache_df)
    st.markdown(
        html, 
        unsafe_allow_html=True
    )

    # Relation force-Vitesse
    f"*Relation force-vitesse (3)*"
    fig = px.scatter(
        x = df_essai_3["VitAng"],
        y = df_essai_3["Force"],
        color = df_essai_3["essai"],
        trendline = 'ols',trendline_color_override="lightgray", trendline_scope = "overall",
        color_discrete_sequence = ["Red","Blue"])

    fig.update_yaxes(rangemode="tozero")

    fig.update_traces(
        marker=dict(
            size=10
        )
    )

    fig.update_layout(
        xaxis_title="Vitesse (rad.s-1)",
        yaxis_title="Force (N)",
        showlegend=False,
        font=dict(
            family="Verdana",
            size=12,
            color="Black"
        )
    )

    st.plotly_chart(fig, use_container_width=True)


    # CSS BUTTON
    m = st.markdown("""
                <style>
                div.stButton > button:first-child {
                    background:linear-gradient(to bottom, #6497b1 5%, #6497b1 100%);
                    background-color:#6497b1;
                    border-radius:14px;
                    # display:inline-block;
                    margin: auto;
                    display :block;
                    cursor:pointer;
                    color:#ffffff;
                    font-family:Arial;
                    font-size:17px;
                    padding:12px 25px;
                    text-decoration:none;
                    text-shadow:0px 1px 0px #000000;
                }
                .myButton:hover {
                    background:linear-gradient(to bottom, #6497b1 5%, #6497b1 100%);
                    background-color:#6497b1;
                }
                .myButton:active {
                    position:relative;
                    top:1px;
                }
                </style>""", unsafe_allow_html=True)

    # DATAFRAME FINALE 
    new_df = pd.DataFrame(columns = ["NOM","A_1","B_1","R2_1","A_2","B_2","R2_2","A_3","B_3","R2_3","Pmax_1","Pmax_2","Pmax_3"])
    if 'df' not in st.session_state:
        st.session_state.df = new_df
        
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Ajoutez les données"):
            new_value = pd.DataFrame([{
                "NOM":filename.name[:-6],
                # "TEST":selected_filename,
                "A_1":a_mod_1,
                "B_1":b_mod_1,
                "R2_1":r2_mod_1,
                "A_2":a_mod_2,
                "B_2":b_mod_2,
                "R2_2":r2_mod_2,
                "A_3":a_mod_3,
                "B_3":b_mod_3,
                "R2_3":r2_mod_3,
                "Pmax_1":Pmax_mod_1,
                "Pmax_2":Pmax_mod_2,
                "Pmax_3":Pmax_mod_3,
            }])
            st.session_state.df = pd.concat([st.session_state.df,new_value])
            st.success('Données ajoutées !', icon="✅")

    with col2:
        if st.button("Suppression dernière ligne"):
            st.session_state.df = st.session_state.df[:-1]
            st.warning('Suppression de la dernière ligne', icon="⚠️")


    with col3:
        if st.button("Supprimer le dataframe"):
            st.session_state.df = st.session_state.df.iloc[0:0]
            st.error('Suppression du dataframe avec succès', icon="🚨")

    st.dataframe(st.session_state.df, use_container_width = True)

    d = st.markdown("""
                <style>
                div.stDownloadButton > button:first-child {
                background:linear-gradient(to bottom, #64b1a5 5%, #64b1a5 100%);
                background-color:#64b1a5;
                border-radius:14px;
                # display:inline-block;
                margin: auto;
                display :block;
                cursor:pointer;
                color:#ffffff;
                font-family:Verdana;
                font-size:17px;
                padding:12px 25px;
                text-decoration:none;
                text-shadow:0px 1px 0px #000000;
            }
            .button:hover {
                background:linear-gradient(to bottom, #64b1a5 5%, #64b1a5 100%);
                background-color:#64b1a5;
            }
            .button:active {
                position:relative;
                top:1px;
            }
                </style>""", unsafe_allow_html=True)
    st.download_button(
        label="Download data as CSV",
        data=st.session_state.df.to_csv(index=False).encode('utf-8'),
        file_name = "analyse_profil_force_vitesse.csv"
    )