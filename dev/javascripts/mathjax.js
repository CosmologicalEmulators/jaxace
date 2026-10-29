window.MathJax = {
  startup: {
    ready() {
      MathJax.startup.defaultReady();
      // Material replaces the document body during instant navigation.
      // Subscribe only after MathJax is ready, and serialize typesetting.
      let pending = MathJax.startup.promise;
      document$.subscribe(() => {
        pending = pending.then(() => {
          MathJax.startup.output.clearCache();
          MathJax.typesetClear();
          MathJax.texReset();
          return MathJax.typesetPromise();
        }).catch(error => console.error("MathJax typesetting failed", error));
      });
    }
  },
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};
