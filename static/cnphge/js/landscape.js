/* Interactive replay of the computed, verified gradient descent. */
(() => {
  const trace = window.CNPHGE_OPTIMIZATION.trajectory;
  const slider = document.getElementById('descent-iteration');
  const format = n => n.toLocaleString('fr-FR',{maximumFractionDigits:2});
  const xy = ([p,a]) => [75+(p-4)/16*555,365-a/180*310];
  function render() {
    const t = Number(slider.value), [p,a,j] = trace[t];
    document.getElementById('descent-path').setAttribute('d',trace.slice(0,t+1).map((v,i)=>(i?'L':'M')+xy(v).join(',')).join(' '));
    const point = document.getElementById('descent-point'), [x,y] = xy(trace[t]);
    point.setAttribute('cx',x);point.setAttribute('cy',y);
    document.getElementById('descent-status').textContent = `${t} · p = ${format(p)} € · a = ${format(a)} € · J = ${format(j)} €`;
    slider.setAttribute('aria-valuetext',`Itération ${t}, prix ${format(p)} euros, budget ${format(a)} euros, coût ${format(j)} euros`);
  }
  slider.addEventListener('input',render);
  slider.addEventListener('keydown',event=>event.stopPropagation());
  render();
})();
