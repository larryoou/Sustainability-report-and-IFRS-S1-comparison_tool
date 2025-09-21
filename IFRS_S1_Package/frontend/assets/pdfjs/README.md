# pdf.js Offline Assets

Place the following files here for offline use inside WKWebView:

- pdf.min.js
- pdf.worker.min.js

You can download a matching version from CDN or release builds, for example:
- https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js
- https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js

In your HTML, replace the CDN references with:

```html
<script src="assets/pdfjs/pdf.min.js"></script>
<script>
  pdfjsLib.GlobalWorkerOptions.workerSrc = 'assets/pdfjs/pdf.worker.min.js';
</script>
```

When bundling into Xcode, include `assets/pdfjs/` inside `Resources/html/` so `WKWebView.loadFileURL` can access it.
