// ---------------------------------------------
// code.ts  —  main thread of the Figma plugin
// ---------------------------------------------

figma.showUI(__html__); // default size is fine for now

// ---------- Types sent from ui.html ----------
interface CreateMsg {
  type: 'create-shapes';
  count: number;
}
interface SendMsg {
  type: 'send';
  count: number;          // still comes through, but we ignore it here
}
interface CancelMsg {
  type: 'cancel';
}
type UIMessage = CreateMsg | SendMsg | CancelMsg;

// ---------- Helper ----------
function getSelectedFrame() {
  const node = figma.currentPage.selection[0];
  if (node && node.type === 'FRAME') {
    return node as FrameNode;
  }
  return null;
}

/**
 * Return an array of { id, name, x, y, width, height } for every direct child
 * of the given frame.  Coordinates are **relative to the frame’s top-left**.
 */
function getChildBBoxes(frame: FrameNode) {
  const fx = frame.absoluteTransform[0][2];
  const fy = frame.absoluteTransform[1][2];

  return frame.children.map(node => {
    const ax = node.absoluteTransform[0][2];
    const ay = node.absoluteTransform[1][2];
    return {
      id: node.id,
      name: node.name,
      x: ax - fx,
      y: ay - fy,
      width: node.width,
      height: node.height
    };
  });
}

// ---------- Message handler ----------
figma.ui.onmessage = (msg: UIMessage) => {
  switch (msg.type) {
    /* 1. original rectangle demo --------------------------- */
    case 'create-shapes': {
      const nodes: SceneNode[] = [];
      for (let i = 0; i < msg.count; i++) {
        const rect = figma.createRectangle();
        rect.x = i * 150;
        rect.fills = [{ type: 'SOLID', color: { r: 1, g: 0.5, b: 0 } }];
        figma.currentPage.appendChild(rect);
        nodes.push(rect);
      }
      figma.currentPage.selection = nodes;
      figma.viewport.scrollAndZoomIntoView(nodes);
      figma.closePlugin();          // demo done
      break;
    }

    /* 2. NEW: return selected-frame size ------------------- */
    case 'send': {
      const frame = getSelectedFrame();
      if (!frame) {
        figma.notify('Please select a frame first');
        figma.ui.postMessage({ type: 'frame-size', error: 'no-frame' });
        return;
      }

      const objects = getChildBBoxes(frame);
      console.log('[CONTROLLER] objects', objects);

      // ▶︎ This line prints frame size
      console.log(`[CONTROLLER] Frame size: ${frame.width} × ${frame.height}`);

      figma.ui.postMessage({
        type: 'frame-size',
        width: frame.width,
        height: frame.height
      });
      break;
    }


    /* 3. user clicked Cancel ------------------------------- */
    case 'cancel': {
      figma.closePlugin();
      break;
    }
  }
};
