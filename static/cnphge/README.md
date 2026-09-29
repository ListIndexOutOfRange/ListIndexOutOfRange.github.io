# Présentation CNPHGE

Deck Reveal.js en français : 29 diapositives visibles, puis 6 annexes. La diapositive « Comment construire une échelle commune ? » est conservée avec `data-visibility="hidden"`.

- URL Hugo : `/cnphge/`
- Aperçu : `hugo server`, puis `http://localhost:1313/cnphge/`.
- Navigation : flèches, `Échap` pour la vue d’ensemble, `S` pour les notes orateur.
- Impression PDF : ouvrir `/cnphge/?print-pdf`, puis imprimer en paysage sans en-têtes ni pieds de page du navigateur.
- Contenu : `index.html`. Style : `css/deck.css`.
- Démonstration interactive : `js/optimization.js` ; sélection L1/L2 et curseur de pente, sur les mêmes sept points synthétiques que les diapositives de régression.
- Reveal et notes orateur : ressources locales partagées dans `static/camma/_reveal/`. Aucun CDN nécessaire.

Le contenu suit `inter_specialty_optimization_presentation_plan.md`. Les graphiques de régression utilisent des données synthétiques et des ajustements calculés. Les graphes de spécialités sont illustratifs. Les sept rapports Gastro–Pneumo proviennent du plan, sans validation du tableau original.

Les annexes E et F identifient explicitement les pièces originales CNAM/SFED manquantes. Les références bibliographiques complètes et les documents sources doivent être ajoutés lorsqu’ils seront disponibles. La convexité est présentée sous les hypothèses de la formulation décrite, sans présumer la formulation exacte de l’implémentation française.

Validation : compilation Hugo et existence des ressources locales vérifiées. Refonte visuelle : couverture et conclusion bleu nuit, hiérarchie typographique, tableaux et conclusions harmonisés. Compteur en bas à gauche, commandes Reveal en bas à droite. Rendu contrôlé dans le navigateur sur les principaux types de diapositives et vérification des limites du contenu sur les 35 diapositives avant l’ajout du plan.
