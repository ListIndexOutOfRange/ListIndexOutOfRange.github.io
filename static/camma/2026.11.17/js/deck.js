Reveal.initialize({
  hash: true,
  slideNumber: false,
  progress: true,
  center: true,
  width: 960,
  height: 700,
  plugins: [RevealNotes]
});
(function () {
  const box = document.getElementById('custom-slide-number');
  function updateNumber() {
    const slide = Reveal.getCurrentSlide();
    if (!slide) return;
    box.hidden = slide.hasAttribute('data-no-number');
    const slides = Reveal.getSlides();
    const group = slide.hasAttribute('data-backup')
      ? slides.filter(s => s.hasAttribute('data-backup'))
      : slides.filter(s => !s.hasAttribute('data-no-count') && !s.hasAttribute('data-backup'));
    const number = group.indexOf(slide) + 1;
    box.textContent = slide.hasAttribute('data-backup') ? `B${number}` : `${number}`;
  }
  Reveal.on('ready', updateNumber);
  Reveal.on('slidechanged', updateNumber);
})();
