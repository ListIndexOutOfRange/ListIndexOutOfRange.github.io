/* Synthetic data shared with the regression slides. No external dependencies. */
(() => {
  const points = [1.1, 1.9, 3.2, 3.8, 5.1, 6, 12].map((y, i) => ({x: i + 1, y}));
  const slider = document.getElementById('slope');
  const selector = document.getElementById('criterion');
  const format = n => n.toLocaleString('fr-FR', {maximumFractionDigits: 2, minimumFractionDigits: 2});
  const cost = (a, criterion) => points.reduce((sum, p) => sum + (criterion === 'L2' ? (p.y-a*p.x)**2 : Math.abs(p.y-a*p.x)), 0);
  const l2 = points.reduce((sum,p)=>sum+p.x*p.y,0) / points.reduce((sum,p)=>sum+p.x*p.x,0);
  const sorted = [...points].sort((p,q)=>p.y/p.x-q.y/q.x);
  let weight = 0;
  const l1 = sorted.find(p => (weight += p.x) >= points.reduce((s,p)=>s+p.x,0)/2);
  const optimum = {L2:l2, L1:l1.y/l1.x};
  const line = (x1,y1,x2,y2,color='#c3d1d7',extra='') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="2" ${extra}/>`;
  const text = (x,y,value,color='#526777',extra='') => `<text x="${x}" y="${y}" fill="${color}" font-size="18" ${extra}>${value}</text>`;
  const circle = (x,y,color,r=6) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${color}"/>`;
  function render() {
    const a = Number(slider.value), criterion = selector.value;
    const px = x => 52 + x*68, py = y => 282-y*14;
    let fit = line(52,282,565,282)+line(52,282,52,18)+text(570,307,'x')+text(22,22,'y');
    for (const x of [0,2,4,6]) fit += text(px(x),307,x,'#526777','text-anchor="middle"');
    for (const y of [5,10,15]) fit += line(48,py(y),565,py(y),'#e4ebee')+text(40,py(y)+6,y,'#526777','text-anchor="end"');
    points.forEach(p => {fit += line(px(p.x),py(p.y),px(p.x),py(a*p.x),'#bd613b','stroke-dasharray="5 4"');});
    fit += line(px(0),py(0),px(7.4),py(a*7.4),'#376bb3','style="stroke-width:3"');
    points.forEach(p=>{fit+=circle(px(p.x),py(p.y),p.x===7?'#bd613b':'#22364a');});
    document.getElementById('fit-plot').innerHTML = fit;
    const max = Math.ceil(Math.max(cost(0,criterion),cost(2.5,criterion))/10)*10;
    const cx = a => 58+a/2.5*500, cy = c => 282-c/max*245;
    let curve = line(58,282,565,282)+line(58,282,58,18)+text(582,307,'a')+text(12,20,'J(a)');
    for (const x of [0,.5,1,1.5,2,2.5]) curve += text(cx(x),307,String(x).replace('.',','),'#526777','text-anchor="middle"');
    for (const y of [0,max/2,max]) curve += line(58,cy(y),558,cy(y),'#e4ebee')+text(48,cy(y)+6,y,'#526777','text-anchor="end"');
    // Include every L1 breakpoint so the piecewise-linear curve is exact.
    const samples = [...Array.from({length:251},(_,i)=>i/100), ...points.map(p=>p.y/p.x)].sort((a,b)=>a-b);
    curve += `<polyline points="${samples.map(a=>`${cx(a)},${cy(cost(a,criterion))}`).join(' ')}" fill="none" stroke="#376bb3" stroke-width="3"/>`;
    const best=optimum[criterion];
    curve += line(cx(a),282,cx(a),cy(cost(a,criterion)),'#bd613b','stroke-dasharray="5 4"');
    curve += circle(cx(best),cy(cost(best,criterion)),'#087e83',7)+circle(cx(a),cy(cost(a,criterion)),'#bd613b',8);
    document.getElementById('cost-plot').innerHTML=curve;
    document.getElementById('slope-value').textContent=format(a);
    slider.setAttribute('aria-valuetext',format(a));
    document.getElementById('cost-title').textContent=criterion==='L2'?'Le critère J₂(a) = ∑ rᵢ²':'Le critère J₁(a) = ∑ |rᵢ|';
    document.getElementById('cost-value').textContent=`● Pente choisie : a = ${format(a)} · J(a) = ${format(cost(a,criterion))}`;
    document.getElementById('minimum-value').textContent=`● Minimum ${criterion} : a ≈ ${format(best)} · J ≈ ${format(cost(best,criterion))}`;
  }
  slider.addEventListener('input', render);
  selector.addEventListener('change', render);
  // Let arrow keys operate form controls without advancing Reveal slides.
  for (const control of [slider,selector]) control.addEventListener('keydown',e=>e.stopPropagation());
  render();
})();
