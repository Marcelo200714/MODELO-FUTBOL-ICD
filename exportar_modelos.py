# ══════════════════════════════════════════════════════════════════════════════
# exportar_modelos.py  —  pega este código al final de tu notebook y ejecútalo
# ══════════════════════════════════════════════════════════════════════════════
import pickle, os

SAVE_DIR = '.'   # misma carpeta que el notebook; cambia si prefieres otra ruta

def guardar(obj, nombre):
    with open(os.path.join(SAVE_DIR, nombre), 'wb') as f:
        pickle.dump(obj, f)
    print(f'  ✅ {nombre} guardado')

print('Exportando modelos...')

# Las 5 variables que realmente usa tu notebook:
guardar(model_home,   'model_home.pkl')
guardar(model_away,   'model_away.pkl')
guardar(df_final,     'df_final.pkl')
guardar(forma,        'forma.pkl')
guardar(player_stats, 'player_stats.pkl')   # <-- SIN elo_dict (no existe en tu notebook)

print('\n✅ Listo. Ahora ejecuta:  python app.py')
