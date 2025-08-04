// ---------------------------------------------
// code.ts  —  main thread of the Figma plugin
// ---------------------------------------------

figma.showUI(__html__, { width: 375, height: 560 });

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
      // @ts-ignore: selectedOptions is sent from UI
      const selectedOptions = (msg as any).selectedOptions || [];
      console.log('[CONTROLLER] Options selected by user:', selectedOptions);
      if (!selectedOptions || selectedOptions.length === 0) {
        figma.notify('Please select at least one option.');
        figma.ui.postMessage({ type: 'frame-size', error: 'no-options' });
        figma.ui.postMessage({ type: 'loading', loading: false }); // Reset UI
        return;
      }
      const frame = getSelectedFrame();
      if (!frame) {
        figma.notify('Please select a frame first');
        figma.ui.postMessage({ type: 'frame-size', error: 'no-frame' });
        figma.ui.postMessage({ type: 'loading', loading: false }); // Reset UI
        return;
      }
      const objects = getChildBBoxes(frame);
      console.log('[CONTROLLER] objects', objects);
      console.log(`[CONTROLLER] Frame size: ${frame.width} × ${frame.height}`);
      // Show loading notification with a long timeout so it stays visible
      figma.notify('Generating heatmaps...', { timeout: 10000 });
      figma.ui.postMessage({ type: 'loading', loading: true }); // Show spinner in UI
      // Send frame and child info to backend server for prediction
      fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frame: {
            id: frame.id,
            name: frame.name,
            width: frame.width,
            height: frame.height
          },
          children: objects,
          selectedOptions
        })
      })
      .then(res => res.json())
      .then(async (result) => {
        // Helper to decode base64 to Uint8Array
        function base64ToBytes(b64: string): Uint8Array {
          return figma.base64Decode(b64);
        }
        for (const [field, b64] of Object.entries(result)) {
          const bytes = base64ToBytes(b64 as string);
          const image = figma.createImage(bytes);
          // Insert as a rectangle in the same position/size as the frame
          const rect = figma.createRectangle();
          rect.resizeWithoutConstraints(frame.width, frame.height);
          rect.x = frame.x;
          rect.y = frame.y;
          rect.fills = [{ type: 'IMAGE', scaleMode: 'FILL', imageHash: image.hash }];
          rect.name = field; // Set the rectangle's name to the field name
          rect.opacity = 0.5; // Set opacity to 50%
          figma.currentPage.appendChild(rect);
        }
        figma.notify('Heatmap(s) inserted!');
        figma.ui.postMessage({ type: 'loading', loading: false }); // Hide spinner
      })
      .catch(err => {
        console.error('Failed to fetch prediction or insert image:', err);
        figma.notify('Failed to fetch prediction or insert image.');
        figma.ui.postMessage({ type: 'loading', loading: false }); // Hide spinner
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
