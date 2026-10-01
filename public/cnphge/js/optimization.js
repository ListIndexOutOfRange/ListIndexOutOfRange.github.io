/* Pricing examples from optimization_pricing_example.md. All charts use the
   actual functions, and all iterations use their analytic derivatives. */
(() => {
  'use strict';
  const color = {ink:'#172e40', muted:'#5b6e79', grid:'#d5e0e3', blue:'#376bb3', teal:'#087e83', purple:'#6656aa', orange:'#b35c36'};
  const format = (n, digits=0) => n.toLocaleString('fr-FR', {minimumFractionDigits:digits, maximumFractionDigits:digits});
  const demand = p => 100 - 5*p;
  const profit = p => (p-4)*demand(p);
  const criterion = p => -profit(p);
  const text = (x,y,label,fill=color.muted,anchor='start',size=22) => `<text x="${x}" y="${y}" fill="${fill}" text-anchor="${anchor}" font-size="${size}">${label}</text>`;
  const line = (x1,y1,x2,y2,stroke=color.grid,dash='') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="1.5" ${dash ? `stroke-dasharray="${dash}"`:''}/>`;
  const dot = (x,y,fill,r=7) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="#f6f8f8" stroke-width="2.5"/>`;
  function chart(id,{xmin,xmax,ymin,ymax,xticks,yticks,xlabel='Prix p (€)',height=420,right=28}) {
    const el = document.getElementById(id);
    const box = {left:75,top:30,right:720-right,bottom:height-70};
    const x = v => box.left+(v-xmin)/(xmax-xmin)*(box.right-box.left);
    const y = v => box.bottom-(v-ymin)/(ymax-ymin)*(box.bottom-box.top);
    let content = '';
    for (const v of yticks) content += line(box.left,y(v),box.right,y(v)) + text(box.left-13,y(v)+7,format(v),color.muted,'end',21);
    content += line(box.left,box.top,box.left,box.bottom)+line(box.left,box.bottom,box.right,box.bottom);
    for (const v of xticks) content += line(x(v),box.bottom,x(v),box.bottom+7)+text(x(v),box.bottom+32,format(v),color.muted,'middle',21);
    content += text(box.right,height-7,xlabel,color.muted,'end',22);
    return {
      x,y,box,
      add: markup => {content += markup;},
      curve(fn,stroke,dash='') {
        // Sampling includes the L1 breakpoint p=10 exactly for the comparison.
        const values = Array.from({length:481},(_,i)=>xmin+(xmax-xmin)*i/480);
        content += `<polyline points="${values.map(v=>`${x(v).toFixed(2)},${y(fn(v)).toFixed(2)}`).join(' ')}" fill="none" stroke="${stroke}" stroke-width="4" stroke-linejoin="round" ${dash?`stroke-dasharray="${dash}"`:''}/>`;
      },
      render: () => {el.innerHTML = content;}
    };
  }
  function drawPriceChart(id, fn, max, ticks, p, stroke, showOptimum, showCost=false) {
    const x = v => 58+(v-4)/16*352, y = v => showCost ? 151-v/max*109 : 260-v/max*218;
    if (showCost) ticks = [-300,0,300];
    let svg = '';
    for (const v of ticks) svg += line(58,y(v),410,y(v))+text(48,y(v)+6,format(v),color.muted,'end',19);
    svg += line(58,42,58,260)+line(58,260,410,260);
    for (const v of [4,8,12,16,20]) svg += text(x(v),286,v,color.muted,'middle',19);
    svg += text(410,315,'Prix (€)',color.muted,'end',18);
    svg += `<polyline points="${Array.from({length:161},(_,i)=>{const v=4+i/10;return `${x(v)},${y(fn(v))}`;}).join(' ')}" fill="none" stroke="${stroke}" stroke-width="3"/>`;
    if (showCost) {
      svg += line(58,y(0),410,y(0),color.muted);
      svg += `<polyline points="${Array.from({length:161},(_,i)=>{const v=4+i/10;return `${x(v)},${y(-fn(v))}`;}).join(' ')}" fill="none" stroke="${color.orange}" stroke-width="3" stroke-dasharray="7 4"/>`;
      svg += dot(x(p),y(-fn(p)),color.orange,6);
      svg += text(410,31,'Profit',stroke,'end',19)+text(410,251,'Coût = −profit',color.orange,'end',19);
    }
    if(showOptimum) svg += line(x(12),260,x(12),y(fn(12)),color.teal,'4 4')+dot(x(12),y(fn(12)),color.teal,7)+text(x(12),y(fn(12))-17,'Maximum',color.teal,'middle',19);
    svg += line(x(p),showCost?y(-fn(p)):260,x(p),y(fn(p)),color.orange,'4 4')+dot(x(p),y(fn(p)),color.orange,7);
    document.getElementById(id).innerHTML=svg;
  }
  function renderPrices(p) {
    for(const prefix of ['pricing-decision','pricing-profit']) {
      const slider=document.getElementById(prefix+'-price'); if (!slider) continue; slider.value=p;
      slider.setAttribute('aria-valuetext',`${format(p,1)} euros`);
      document.getElementById(prefix+'-price-value').textContent=`${format(p,1)} €`;
      for(const [key,fn,max,ticks,stroke,unit] of [
        ['demand',demand,85,[0,20,40,60,80],color.blue,'unités'],
        ['margin',p=>p-4,18,[0,4,8,12,16],color.purple,'€ / unité'],
        ['profit',profit,380,[0,100,200,300],color.teal,'€ de profit']]) {
        drawPriceChart(prefix+'-'+key,fn,max,ticks,p,stroke,prefix==='pricing-profit'&&key==='profit',key==='profit');
        document.getElementById(prefix+'-'+key+'-value').textContent=`${format(fn(p),key==='profit'?2:1)} ${unit}`;
      }
    }
  }
  const compare = chart('criteria-plot',{xmin:8,xmax:14,ymin:-330,ymax:-80,xticks:[8,9,10,11,12,13,14],yticks:[-300,-200,-100]});
  compare.add(line(compare.x(10),compare.box.top,compare.x(10),compare.box.bottom,color.muted,'4 5'));
  compare.add(text(compare.x(10)+12,compare.box.top+18,'Référence : 10 €',color.muted,'start',21));
  for (const [index,[fn,best,stroke,dash]] of [[criterion,12,color.blue,''],[p=>criterion(p)+10*Math.abs(p-10),11,color.teal,'10 5'],[p=>criterion(p)+10*(p-10)**2,32/3,color.purple,'3 5']].entries()) {
    compare.add(`<g class="fragment" data-fragment-index="${index}">`);
    compare.curve(fn,stroke,dash);
    compare.add(dot(compare.x(best),compare.y(fn(best)),stroke,8));
    compare.add('</g>');
  }
  compare.render();
  // Include SVG groups in Reveal’s fragment ordering after chart generation.
  if (window.Reveal) Reveal.sync();
  document.querySelectorAll('.shared-price').forEach(slider=>{
    slider.addEventListener('input',()=>renderPrices(Number(slider.value)));
    slider.addEventListener('keydown',event=>event.stopPropagation());
  });
  renderPrices(8);
})();
