# Présentation CNPHGE

Deck Reveal.js en français : 30 diapositives visibles, puis 6 annexes. La diapositive « Comment construire une échelle commune ? » est conservée avec `data-visibility="hidden"`.

- URL Hugo : `/cnphge/`
- Aperçu : `hugo server`, puis `http://localhost:1313/cnphge/`.
- Navigation : flèches, `Échap` pour la vue d’ensemble, `S` pour les notes orateur.
- Impression PDF : ouvrir `/cnphge/?print-pdf`, puis imprimer en paysage sans en-têtes ni pieds de page du navigateur.
- Contenu : `index.html`. Style : `css/deck.css`.
- Introduction à l’optimisation : 8 diapositives fondées sur `optimization_pricing_example.md`, de la décision de prix à la convexité et au diagnostic modèle/calcul.
- Graphiques et démonstration : `js/optimization.js`. Le curseur de prix actualise les ventes et le profit. Les courbes L1/L2, les minima et les trajectoires de descente de gradient proviennent des fonctions de l’exemple. Aucun service externe nécessaire.
- Reveal et notes orateur : ressources locales partagées dans `static/camma/_reveal/`. Aucun CDN nécessaire.

L’introduction suit `optimization_pricing_example.md`. La suite conserve la structure de `inter_specialty_optimization_presentation_plan.md`. La comparaison des critères utilise des poids illustratifs λ₁ = 10 et λ₂ = 10 (unités différentes), donnant des prix de 12 €, 11 € et 10,67 €. Le modèle à trois pics est un exemple pédagogique non monotone, pas une demande de marché calibrée. La descente de gradient utilise un pas de 0,008, à partir de 7 €, 14 € et 20 €. Les détails figurent dans les notes orateur. L’exemple de régression conservé dans la partie sur la sensibilité est indépendant et synthétique. Les graphes de spécialités sont illustratifs. Les sept rapports Gastro–Pneumo proviennent du plan, sans validation du tableau original.

Les annexes E et F identifient explicitement les pièces originales CNAM/SFED manquantes. Les références bibliographiques complètes et les documents sources doivent être ajoutés lorsqu’ils seront disponibles. La convexité est présentée sous les hypothèses de la formulation décrite, sans présumer la formulation exacte de l’implémentation française.

Validation : introduction inspectée visuellement dans le navigateur, curseur et navigation clavier vérifiés. Compteur calculé à partir du nombre de diapositives principales, annexes identifiées séparément. Les graphiques sont des SVG produits localement et restent disponibles à l’impression depuis le navigateur.
