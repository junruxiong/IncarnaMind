/* Stands in for office.js in a headless browser: a document the pane can read and
   write, shaped like Word's and Excel's APIs (only the calls the pane makes).
   The run's settings are in window.__fake: host, platform, paragraphs, cells. */
(() => {
  const fake = window.__fake;
  fake.comments = [];
  fake.notes = [];

  const range = (text, start) => ({
    text,
    start,
    end: start + text.length,
    expandTo(other) {
      return range(fake.text.slice(start, other.end), start);
    },
    insertComment(comment) {
      fake.comments.push({ on: text, comment });
    },
  });

  fake.text = fake.paragraphs.join("\n");

  const word = {
    run: async (callback) =>
      callback({
        sync: async () => undefined,
        document: {
          body: {
            paragraphs: {
              load: () => undefined,
              items: fake.paragraphs.map((text) => ({ text })),
            },
            search: (query) => {
              const needle = query.replaceAll("^^", "^");
              const items = [];
              for (let at = fake.text.indexOf(needle); at >= 0; at = fake.text.indexOf(needle, at + 1)) {
                items.push(range(needle, at));
              }
              return { load: () => undefined, items };
            },
          },
        },
      }),
  };

  const excel = {
    run: async (callback) => {
      const selected = {
        values: fake.values,
        formulas: fake.formulas,
        address: "Sheet1!A1:B2",
        load: () => undefined,
        getCell: (row, column) => ({ row, column }),
      };
      return callback({
        sync: async () => undefined,
        workbook: {
          getSelectedRange: () => selected,
          worksheets: {
            getActiveWorksheet: () => ({
              notes: {
                add: (cell, text) => fake.notes.push({ row: cell.row, column: cell.column, text }),
              },
            }),
          },
        },
      });
    },
  };

  window.Word = word;
  window.Excel = excel;
  window.Office = {
    HostType: { Word: "Word", Excel: "Excel" },
    PlatformType: { PC: "PC", Mac: "Mac", OfficeOnline: "OfficeOnline" },
    context: {
      diagnostics: { version: "fake" },
      requirements: { isSetSupported: () => true },
    },
    onReady: (callback) => {
      const start = () => callback({ host: fake.host, platform: fake.platform });
      if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
      else start();
    },
  };
})();
