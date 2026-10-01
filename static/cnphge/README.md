# Présentation CNPHGE

Deck Reveal.js en français : 27 diapositives principales, puis 9 annexes. La diapositive « Comment construire une échelle commune ? » reste masquée. Le compteur se calcule automatiquement.

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

Les anciennes slides 18–19 sont réunies dans `specialty-alignment` : quatre étapes avec les flèches, adaptation schématique de Braun (1988), figure 1. Les échelles logarithmiques coulissent sans déformation puis sont superposées ; les petits écarts résiduels sont conservés. `js/alignment.js` gère les transitions, le retour arrière et la réduction des animations.

### Articles de référence fournis

- Hsiao WC, Yntema DB, Braun P, Dunn D, Spencer C. *Measurement and Analysis of Intraservice Work*. JAMA. 1988;260:2361–2370. Enquête auprès de 1 977 médecins de 18 spécialités ; estimation relative au service de référence propre à chaque spécialité. Les quatre dimensions étudiées sont le temps, l’effort mental et le jugement, la technicité et l’effort physique, et le stress. La cohérence et la reproductibilité des jugements ne constituent pas une validation par un étalon objectif ; les coefficients estimés ne sont pas à transposer directement entre spécialités.
- Braun P et al. *Cross-Specialty Linkage of Resource-Based Relative Value Scales: Linking Specialties by Services and Procedures of Equal Work*. JAMA. 1988;260:2390–2396. Liens entre services identiques ou jugés équivalents, sélection clinique et contrôle des temps ; 82 liens finaux et 133 évaluations participant aux liens. Alignement des logarithmes par moindres carrés pondérés, avec positions communes de compromis et fixation d’une origine arbitraire. Les rapports internes sont préservés. L’écart RMS d’environ 7 % décrit l’ajustement aux liens retenus, pas une erreur mesurée face à une vérité externe. Six liens aux écarts extrêmes ont été retirés avant le réseau final ; les analyses de sensibilité rapportées concernent ce processus historique, pas automatiquement l’implémentation française.

La figure animée de la slide 18 reprend le principe qualitatif de la figure 1 de Braun (p. 2392), et non ses coordonnées ni des données cliniques réelles. Les écarts persistants préparent l’explication ultérieure du compromis entre liens.

Les anciennes slides 19–21 sont fusionnées dans `network-objective` : apparition du réseau et des variables, puis des résidus annotés, puis du coût quadratique pondéré. Le schéma distingue explicitement cette forme pédagogique par liens du critère français restant à documenter. Les positions graphiques des nœuds ne représentent pas les variables numériques.

Sur `network-objective`, b désigne un décalage logarithmique. Le résidu signé est r_AB = (b_A − b_B) − c_AB, où c_AB traduit la relation clinique entre les actes liés. Le critère présenté est la somme non pondérée des carrés ; une origine est fixée. La conversion depuis les facteurs multiplicatifs et les limites de cette formulation pédagogique par liens sont précisées dans les notes orateur.


Les slides 20–21 (`optimization-procedure`, `optimization-analysis`) s’appuient sur l’analyse SFED fournie, *CCAM – Tome 2*, T. Ponchon, L. Palazzo, J.-M. Canard, p. 10–11, 17–18 et 22. La procédure rapportée alterne ajustement et retrait de la pire passerelle dépassant un seuil relatif de 20 %. Le dénominateur, le cas d’égalité, le critère exact et les règles d’ex æquo restent à préciser. Le seuil de 25 % sur les durées est distinct. L’analyse sépare la convexité du modèle quadratique à réseau fixé de la sélection itérative des liens ; elle ne prétend pas démontrer la convexité du logiciel français. Les résidus de la slide 19 utilisent uniformément les décalages logarithmiques.

Une slide `graph-consistency` suit le réseau (slide 20) : triangle GP–PK–GK, cohérence des c sur les cycles et compromis sur les résidus. La procédure et son analyse passent aux slides 21–22. L’étape Ajuster montre les quatre nœuds et cinq liens du réseau, conservés ensemble avant tout retrait.

Fin resserrée aux slides 23–25 : redondance et ponts, qualité et incertitude des liens, transparence. Les comparaisons L2/L1, les biais partagés et les six tests de sensibilité sont déplacés aux annexes G–I. Le constat de transparence est attribué au document SFED historique, sans affirmer un état actuel non vérifié.

La slide 26 (`take-home-message`) conclut avec les trois composantes de l’optimisation, les risques liés au critère et les exigences de redondance indépendante, d’incertitude et de transparence.

L’intertitre de troisième partie `section-interpretation` est placé après la slide 22 : « Interprétation — Forces et limites de la procédure ».

L’intertitre de troisième partie précède désormais l’analyse de convexité : intertitre en slide 22, analyse en slide 23.
