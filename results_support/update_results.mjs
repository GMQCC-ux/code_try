import fs from 'node:fs/promises';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

const request = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(request.workbook));
const sheet = workbook.worksheets.getItemAt(0);
if (request.inspectOnly) {
  console.log((await workbook.inspect({kind:'table',range:`${sheet.name}!A1:E6`,include:'values',tableMaxRows:6,tableMaxCols:5})).ndjson);
  const image = await workbook.render({sheetName:sheet.name,range:'A1:E6',scale:1.5,format:'png'});
  await fs.writeFile(request.preview, new Uint8Array(await image.arrayBuffer()));
  process.exit(0);
}

const fields = [
  ['数据集',request.dataset], ['轮次',request.epochs],
  ['DICE',request.test.dice], ['Iou',request.test.iou],
  ['测试HD95',request.test.hd95], ['测试Precision',request.test.precision], ['测试Recall',request.test.recall],
  ['验证Dice',request.validation.dice], ['验证IoU',request.validation.iou],
  ['验证HD95',request.validation.hd95], ['验证Precision',request.validation.precision], ['验证Recall',request.validation.recall],
  ['Batch size',request.batchSize], ['随机种子',request.seed],
  ['Checkpoint',request.checkpoint], ['实验标识',request.runId], ['完成时间',request.finishedAt],
];
const used = sheet.getUsedRange().values;
const headers = [...used[0]];
for (const [name] of fields) {
  if (!headers.includes(name)) headers.push(name);
}
let rowIndex = used.findIndex((row,index)=>index>0 && row[headers.indexOf('实验标识')]===request.runId);
if (rowIndex < 0 && request.matchExisting) {
  const candidates = used.map((row,index)=>({row,index})).filter(({row,index}) => index>0 &&
    !row[headers.indexOf('实验标识')] && row[headers.indexOf('数据集')]===request.dataset &&
    Number(row[headers.indexOf('轮次')])===request.epochs &&
    Math.abs(Number(row[headers.indexOf('DICE')])-request.test.dice)<1e-8);
  if (candidates.length!==1) throw new Error('Existing-row match must be unique; workbook was not changed');
  rowIndex=candidates[0].index;
}
const isNew = rowIndex<0;
if (isNew) rowIndex=used.length;
sheet.getRangeByIndexes(0,0,1,headers.length).values=[headers];
// Update only automatic columns. Never write to 模型 or other manual columns.
for (const [name,value] of fields) {
  const col=headers.indexOf(name);
  const cell=sheet.getCell(rowIndex,col);
  if (isNew && rowIndex>1) cell.copyFrom(sheet.getCell(rowIndex-1,col),'all');
  cell.values=[[value]];
  if (['DICE','Iou','测试Precision','测试Recall','验证Dice','验证IoU','验证Precision','验证Recall'].includes(name))
    cell.setNumberFormat('0.00%');
  if (name.endsWith('HD95')) cell.setNumberFormat('0.0000');
  if (['轮次','Batch size','随机种子'].includes(name)) cell.setNumberFormat('0');
}
// Extend widths to fit new headings, keeping the supplied A:E layout.
for(let col=used[0].length;col<headers.length;col++) {
  sheet.getRangeByIndexes(0,col,rowIndex+1,1).format.columnWidth =
    ['Checkpoint','实验标识'].includes(headers[col])?44 : headers[col]==='完成时间'?23:18;
}
workbook.recalculate();
console.log((await workbook.inspect({kind:'table',range:`${sheet.name}!A1:Q${rowIndex+1}`,include:'values',tableMaxRows:8,tableMaxCols:17,maxChars:3500})).ndjson);
if(request.preview) {
  const image=await workbook.render({sheetName:sheet.name,range:`A1:M${Math.min(rowIndex+1,8)}`,scale:1.5,format:'png'});
  await fs.writeFile(request.preview,new Uint8Array(await image.arrayBuffer()));
}
const temporary=request.workbook+'.writing.xlsx';
const output=await SpreadsheetFile.exportXlsx(workbook);
await output.save(temporary);
const handle=await fs.open(temporary,'r+');
await handle.sync();await handle.close();
await fs.rename(temporary,request.workbook);
console.log(`[RESULTS_XLSX] ${isNew?'Added':'Updated'} row ${rowIndex+1}: ${request.workbook}`);
