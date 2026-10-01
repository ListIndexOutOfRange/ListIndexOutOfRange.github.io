/* Qualitative adaptation of Braun et al. (1988), figure 1. */
(() => {
  const slide = document.getElementById('specialty-alignment');
  if (!slide) return;
  const byId = id => document.getElementById(id);
  const stages = [
    ['Chacun sa référence', 'Les actes sont ordonnés au sein de chaque spécialité. Leurs niveaux ne sont pas encore comparables.', 'Deux références différentes, même si chacune vaut 100 dans sa propre spécialité.'],
    ['Des actes jugés équivalents', 'A₁ et A₂, B₁ et B₂, C₁ et C₂ : chaque paire représente un travail jugé comparable.', 'Ces liens donnent des repères pour rapprocher les deux échelles.'],
    ['Rapprocher les actes liés', 'Les échelles coulissent en bloc pour rapprocher au mieux les trois paires.', 'Un compromis : les actes liés ne coïncident pas tous exactement.'],
    ['Lire tous les actes ensemble', 'Les deux spécialités partagent désormais une même échelle de travail.', 'Les actes non liés et les références ont suivi le déplacement de leur spécialité.']
  ];
  let frame = 0, state = [0,0,0];
  function paint([links,shift,merge]) {
    const x1 = 220 + 170*merge, x2 = 560 - 170*merge, dy = 160/3*shift;
    byId('alignment-scale-1').setAttribute('transform',`translate(${x1},${dy})`);
    byId('alignment-scale-2').setAttribute('transform',`translate(${x2},${-dy})`);
    byId('alignment-links').setAttribute('opacity',links*(1-merge));
    slide.querySelectorAll('#alignment-links line').forEach(line => {
      line.setAttribute('x1',x1); line.setAttribute('x2',x2);
      line.setAttribute('y1',Number(line.dataset.y1)+dy);
      line.setAttribute('y2',Number(line.dataset.y2)-dy);
    });
    slide.querySelectorAll('.alignment-rail').forEach(line=>line.setAttribute('opacity',1-merge));
    ['alignment-name-1','alignment-name-2'].forEach(id=>byId(id).setAttribute('opacity',1-merge));
    ['alignment-axis','alignment-common-name'].forEach(id=>byId(id).setAttribute('opacity',merge));
  }
  function update(instant=false, printing=false) {
    cancelAnimationFrame(frame);
    const stage = printing ? 3 : slide.querySelectorAll('.alignment-trigger.visible').length;
    const target=[stage>=1?1:0,stage>=2?1:0,stage>=3?1:0];
    ['alignment-heading','alignment-description','alignment-detail'].forEach((id,i)=>byId(id).textContent=stages[stage][i]);
    slide.querySelectorAll('.alignment-steps span').forEach((el,i)=>el.classList.toggle('active',i===stage));
    if(instant || matchMedia('(prefers-reduced-motion: reduce)').matches) {state=target;paint(state);return;}
    const from=[...state], start=performance.now();
    function tick(now) {
      const t=Math.min(1,(now-start)/1000), ease=t*t*(3-2*t);
      state=target.map((v,i)=>from[i]+(v-from[i])*ease);paint(state);
      if(t<1)frame=requestAnimationFrame(tick);
    }
    frame=requestAnimationFrame(tick);
  }
  Reveal.on('fragmentshown',e=>{if(slide.contains(e.fragment))update();});
  Reveal.on('fragmenthidden',e=>{if(slide.contains(e.fragment))update();});
  Reveal.on('slidechanged',e=>{cancelAnimationFrame(frame);if(e.currentSlide===slide)update(true);});
  Reveal.on('ready',()=>update(true,location.search.includes('print-pdf')));
  window.addEventListener('beforeprint',()=>update(true,true));
  window.addEventListener('afterprint',()=>update(true));
  update(true,location.search.includes('print-pdf'));
})();
