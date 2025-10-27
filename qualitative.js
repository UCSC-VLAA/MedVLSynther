(function () {
  const scrollers = document.querySelectorAll('.qual-scroller[data-animated="true"]');
  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  scrollers.forEach(scroller => {
    if (prefersReduced) {
      scroller.removeAttribute('data-animated');
      return;
    }
    const track = scroller.querySelector('.qual-track');
    if (!track) return;

    const gapPx = parseFloat(getComputedStyle(track).gap || '24');
    const children = Array.from(track.children);
    if (children.length === 0) return;

    // 克隆一遍，左右两半相同，实现无缝循环
    children.forEach(node => {
      const clone = node.cloneNode(true);
      clone.setAttribute('aria-hidden', 'true');
      track.appendChild(clone);
    });

    // 计算总宽度（含 gap），并根据 data-speed(像素/秒) 设定动画时长
    const totalWidth = Array.from(track.children)
      .reduce((sum, el, i, arr) => sum + el.getBoundingClientRect().width + (i ? gapPx : 0), 0);

    const pxPerSec = Math.max(40, parseFloat(scroller.getAttribute('data-speed')) || 120);
    const duration = Math.max(18, totalWidth / pxPerSec);
    track.style.setProperty('--_duration', `${duration}s`);
  });
})();
