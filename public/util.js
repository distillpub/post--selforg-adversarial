export function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    const frame = window.frameElement;
    const frameRect = frame ? frame.getBoundingClientRect() : null;
    const [px, py] = frame ? [frameRect.left, frameRect.top] : [0, 0];
    const w = window.top.innerWidth;
    const h = window.top.innerHeight;
    return rect.top+py < h && rect.left+px < w && rect.bottom+py > 0 && rect.right+px > 0;
  }