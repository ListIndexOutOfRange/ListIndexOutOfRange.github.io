# Présentation CNPHGE

Deck Reveal.js en français : 35 diapositives principales, puis 6 annexes. La diapositive « Comment construire une échelle commune ? » reste masquée. Le compteur se calcule automatiquement.

- URL Hugo : `/cnphge/`.
- Aperçu autonome : `python3 -m http.server 8765 --directory static`, puis `http://localhost:8765/cnphge/`.
- Navigation : flèches, `Échap` pour la vue d’ensemble, `S` pour les notes orateur.
- Impression : ouvrir `/cnphge/?print-pdf`, puis imprimer en paysage sans en-têtes ni pieds de page du navigateur. Les animations affichent leur état final pour l’impression.
- Contenu : `index.html`. Style : `css/deck.css`. Reveal et notes orateur : ressources locales dans `static/camma/_reveal/`.

## Introduction visuelle

Les slides 8 à 13 développent l’exemple du prix et du budget publicitaire, avec peu de texte mathématique à l’écran. Les détails et les limites des modèles sont dans les notes.

- **Deux décisions** : deux colonnes présentent les effets du prix et de la publicité, puis un objectif commun : le profit. Aucun curseur ni graphique sur cette slide.
- **Relief convexe** : une surface 3D présente la hauteur du coût. Il s’agit d’un exemple quadratique pédagogique explicitement simplifié, distinct du modèle logistique précédent. Avec x=(p−12)/8 et y=(a−90)/90, le coût relatif est F=(x²+1,6y²+0,5xy)/2. Son minimum est (12 €, 90 €). Ce coût ne représente pas une prévision monétaire de profit.
- **Descente de gradient** : les anciennes slides sur la recherche, le gradient et les itérations sont réunies. La trajectoire part de (x,y)=(−0,95 ;−0,95), avec un pas de 0,18, sur 32 itérations.
- **Convergence** : le relief et l’historique du coût montrent les mêmes itérations. Des variations faibles ne suffisent pas à certifier l’optimalité.
- **Non-convexité** : le modèle logistique de campagne est conservé. Le creux A est approfondi par un terme gaussien ajouté au modèle (amplitude 180 €), et non par une déformation de sa représentation. Les surfaces utilisent un maillage filaire espacé, avec 13 courbes par direction et une hauteur visuelle renforcée. Six descentes animées rejoignent deux vallées. Le calcul emploie les coordonnées normalisées, le coût J/100, un pas initial de 0,004, une projection sur le domaine et une réduction du pas si le coût augmente. Après 160 itérations, les coûts sont proches des deux minima numériques à moins d’un centime.

Les animations se lancent avec **Lire**. **Pause**, **Recommencer** et le curseur permettent de contrôler le déroulement. Quitter la diapositive ou masquer l’onglet met la lecture en pause. Aucune animation ne démarre automatiquement.

`js/optimization.js` gère l’exemple à un prix et les critères L1/L2. `js/landscape.js` calcule les surfaces, les trajectoires et leur projection SVG en 3D, sans dépendance réseau ni WebGL. Les minima du scénario modifié sont recalculés avec la même fonction que les trajectoires. `js/landscape-data.js` conserve les données du scénario antérieur. Les anciens SVG dans `assets/` restent disponibles comme ressources historiques.

Le modèle de campagne est synthétique. Ses minima sont A ≈ (8,42 € ; 32,73 €), profit illustratif ≈ 256,33 €, et B ≈ (13,03 € ; 132,70 €), profit illustratif ≈ 436,66 €. B est le meilleur minimum numérique identifié sur le domaine p ∈ [4,20] €, a ∈ [0,180] €. Plusieurs initialisations ne constituent pas, à elles seules, une garantie d’optimalité globale.

## Suite de la présentation

La comparaison L1/L2 conserve des poids illustratifs de 10, avec des unités différentes, donnant des prix de 12 €, 11 € et 10,67 €. L’exemple de régression dans la partie sur la sensibilité est indépendant et synthétique. Les graphes de spécialités sont illustratifs. Les sept rapports Gastro–Pneumo proviennent du plan, sans validation du tableau original.

Les annexes E et F signalent les pièces originales CNAM/SFED manquantes. Les références bibliographiques complètes et les documents sources restent à ajouter. La convexité de la hiérarchisation est présentée sous les hypothèses de la formulation décrite, sans présumer la formulation exacte de l’implémentation française.

Les anciennes slides 10 (taille du pas) et 14 (comparaison des points de départ) ont été retirées. Le creux A utilise le terme −180 exp(−(u−uA)²/0,025−(v−vA)²/0,018), centré sur le minimum A du scénario précédent. Les valeurs monétaires restent purement illustratives.

Deux intertitres introduisent les parties « Introduction à l’optimisation » et « Hiérarchisation inter-spécialités ». La conclusion de la première partie distingue un critère inadapté, malgré un minimum global trouvé, d’un échec du solveur à atteindre ce minimum. Elle précède l’intertitre de la deuxième partie.

## Publication pendant la préparation

Le workflow `.github/workflows/hugo.yaml` utilise `CNPHGE_PUBLICATION: title`.
Après la construction Hugo, `scripts/prepare_cnphge_publication.py` remplace uniquement
le dossier généré `public/cnphge` par la couverture et son style, sans les autres
slides, notes ou ressources de travail. L’URL publique reste `/cnphge/`.
Les aperçus locaux conservent le deck complet ; les commits intermédiaires peuvent
être poussés sur `master` sans publier les slides en cours.

Pour publier la présentation complète, changer ce réglage en `full`, puis pousser
sur `master`. Revenir à `title` rétablit la couverture au déploiement suivant.
Ce réglage limite le site publié, pas la visibilité des sources dans le dépôt Git.
