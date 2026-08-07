import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "D:/code/rlGame/.artifact_work/空天地海大模型技术方向及场景收集.xlsx";
const outputDir = "D:/code/rlGame/outputs/019f6eeb-813c-7fe0-947c-dc9ac995ed4d";
const outputPath = `${outputDir}/空天地海大模型技术方向及场景收集_卫星追逃方向已填写.xlsx`;
const previewPath = "C:/Users/liyu/.codex/visualizations/2026/07/17/019f6eeb-813c-7fe0-947c-dc9ac995ed4d/filled_Sheet1.png";

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
const sheet = workbook.worksheets.getItem("Sheet1");

sheet.getRange("A12").values = [["面向在轨追逃博弈的多模态大模型与自主决策验证平台"]];
sheet.getRange("B12").values = [["面向空间目标监视、在轨服务与安全防御等场景，融合光学/雷达观测、遥测轨道时序、航天器状态、任务文本和仿真轨迹等多模态数据，研究空间态势理解、目标行为与意图识别、轨迹预测、追逃策略生成及风险解释。结合轨道动力学模型、强化学习智能体、仿真软件和工具调用能力，构建可进行任务规划、策略推演、人机协同复核和闭环验证的空天博弈大模型平台。"]];
sheet.getRange("A12:H12").format.wrapText = true;
sheet.getRange("A12:H12").format.verticalAlignment = "center";
sheet.getRange("A12").format.horizontalAlignment = "left";
sheet.getRange("B12:H12").format.horizontalAlignment = "left";
sheet.getRange("A12:H12").format.rowHeight = 92;

const check = await workbook.inspect({
  kind: "table",
  range: "Sheet1!A8:H12",
  include: "values,formulas",
  tableMaxRows: 8,
  tableMaxCols: 8,
  maxChars: 6000,
});
console.log(check.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const preview = await workbook.render({
  sheetName: "Sheet1",
  range: "A1:H14",
  scale: 1.5,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

await fs.mkdir(outputDir, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(outputPath);
