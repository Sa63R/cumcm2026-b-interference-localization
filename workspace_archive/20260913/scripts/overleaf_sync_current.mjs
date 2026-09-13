import {Client} from '/Users/zephyrr/.codex/mcp/overleaf-git-mcp/node_modules/@modelcontextprotocol/sdk/dist/esm/client/index.js';
import {StdioClientTransport} from '/Users/zephyrr/.codex/mcp/overleaf-git-mcp/node_modules/@modelcontextprotocol/sdk/dist/esm/client/stdio.js';
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {execFile as execFileCallback} from 'node:child_process';
import {promisify} from 'node:util';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const execFile = promisify(execFileCallback);
const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const packageRoot = '/Users/zephyrr/.codex/mcp/overleaf-git-mcp';
const projectId = '6aa28083141219d8d2d1bdb1';
const repo = path.join(packageRoot, 'repos', projectId);
const staging = path.join(root, 'output/overleaf/项目本地待同步');
const evidence = path.join(root, 'output/overleaf/sync_evidence');
const mode = process.argv[2] || 'prepare';
const hash = text => createHash('sha256').update(text).digest('hex');
const git = async args => (await execFile('/usr/bin/git', ['-c', 'core.quotepath=false', ...args], {cwd:repo})).stdout.trimEnd();
const client = new Client({name:'current-model-sync',version:'1.0.0'});
const transportEnv = process.argv.includes('--http1') ? {
  GIT_CONFIG_COUNT:'1', GIT_CONFIG_KEY_0:'http.version', GIT_CONFIG_VALUE_0:'HTTP/1.1'
} : {};
if(process.argv.includes('--buffer10m')) {
  const index=Number(transportEnv.GIT_CONFIG_COUNT||0);
  transportEnv.GIT_CONFIG_COUNT=String(index+1);
  transportEnv['GIT_CONFIG_KEY_'+index]='http.postBuffer';
  transportEnv['GIT_CONFIG_VALUE_'+index]='10485760';
}
if(process.argv.includes('--trace-safe')) Object.assign(transportEnv,{
  GIT_TRACE_CURL:'1', GIT_TRACE_CURL_NO_DATA:'1', GIT_TRACE_REDACT:'1'
});
if(process.argv.includes('--local-proxy')) {
  const index=Number(transportEnv.GIT_CONFIG_COUNT||0);
  transportEnv.GIT_CONFIG_COUNT=String(index+1);
  transportEnv['GIT_CONFIG_KEY_'+index]='http.proxy';
  transportEnv['GIT_CONFIG_VALUE_'+index]='http://127.0.0.1:7897';
}
const transport = new StdioClientTransport({command:process.execPath,args:[path.join(packageRoot,'overleaf-mcp-server.js')],env:transportEnv,stderr:'pipe'});
const call = async (name, args = {}) => {
  const result = await client.callTool({name,arguments:{projectName:'default',...args}});
  const text = result.content.filter(x=>x.type==='text').map(x=>x.text).join('\n');
  if(result.isError) {
    if(process.argv.includes('--trace-safe')) {
      const lines=text.split('\n').filter(line=>
        /^(error: RPC|send-pack:|fatal:)/.test(line)||
        /[<=>] (?:Send|Recv) header: (?:HTTP\/|GET |POST |Content-Length:|Transfer-Encoding:|Expect:)/i.test(line)||
        /== Info: (?:Trying |Connected to |ALPN:|SSL connection using |upload completely sent off|We are completely uploaded|Recv failure|Closing connection|Connection #|Re-using existing)/.test(line)
      );
      throw new Error('Sanitized connection diagnostic:\n'+lines.join('\n'));
    }
    throw new Error(text);
  }
  return text;
};
const save = async (name, contents) => {
  const target=path.join(evidence,name);
  await mkdir(path.dirname(target),{recursive:true});
  await writeFile(target,contents);
};
const changedFiles = ['main.tex','q1.tex','figures/q1_case.tex','问题一_当前版.tex'];
try {
  await client.connect(transport);
  const projects=JSON.parse(await call('list_projects'));
  if(!projects.some(p=>p.id==='default'&&p.projectId===projectId)) throw new Error('Unexpected project binding');
  if(mode==='prepare') {
    if(await git(['status','--porcelain'])) throw new Error('MCP checkout contains pre-existing local work');
    const files=(await call('list_files')).split('\n').filter(Boolean);
    const snapshots={};
    for(const name of [...new Set([...changedFiles,'q2.tex','问题一_定位区域模型.tex'])]) {
      if(!files.includes(name)) continue;
      const content=await call('read_file',{filePath:name,mode:'full'});
      await save('before/'+name,content);snapshots[name]=hash(content);
    }
    const head=await git(['rev-parse','HEAD']);
    const tree=await git(['ls-tree','-r','HEAD']);
    await save('before_tree.txt',tree+'\n');
    const result={projectId,head,files,snapshots,preparedAt:new Date().toISOString()};
    await save('prepare.json',JSON.stringify(result,null,2)+'\n');
    console.log(JSON.stringify(result,null,2));
  } else if(mode==='push'||mode==='resume'||mode==='retry') {
    const before=JSON.parse(await readFile(path.join(evidence,'prepare.json'),'utf8'));
    if(mode==='push'&&await git(['status','--porcelain'])) throw new Error('MCP checkout contains pre-existing local work');
    await call('list_files');
    if(mode!=='retry'&&await git(['rev-parse','HEAD'])!==before.head) throw new Error('Remote changed after preparation; merge latest before retrying');
    if(mode==='retry') {
      if(await git(['status','--porcelain'])) throw new Error('Unexpected edits after interrupted push');
      const committed=(await git(['diff','--name-only',before.head,'HEAD'])).split('\n').filter(Boolean);
      if(committed.some(name=>!changedFiles.includes(name))) throw new Error('Interrupted commit contains unexpected paths');
    }
    const payload={};
    for(const name of changedFiles) payload[name]=await readFile(path.join(staging,name),'utf8');
    const standalone=payload['问题一_当前版.tex'];
    if(/\\subsection\*\{5\.[45]/.test(standalone)||standalone.includes('楔形张角仅')) throw new Error('Wrong document revision');
    if(!standalone.includes('五、模型的建立与求解')) throw new Error('Missing requested section');
    if(mode==='push') {
      for(const name of changedFiles) console.log(await call('write_file',{filePath:name,content:payload[name]}));
    } else {
      for(const name of changedFiles) {
        if(hash(await readFile(path.join(repo,name),'utf8'))!==hash(payload[name])) throw new Error('Local recovery content differs: '+name);
      }
    }
    const status=await git(['status','--porcelain','-uall']);
    const actual=status.split('\n').filter(Boolean).map(line=>line.slice(3));
    if(actual.some(name=>!changedFiles.includes(name))) throw new Error('Unexpected files would be committed: '+actual.join(', '));
    const diff=(await execFile('/usr/bin/git',['-c','core.quotepath=false','diff','--'],{cwd:repo})).stdout;
    await save('applied.diff',diff);
    const alreadyOnRemote=mode==='retry'&&(await git(['branch','-r','--contains','HEAD'])).includes('origin/');
    console.log(alreadyOnRemote?'Existing commit is already present on Overleaf.':await call('push_changes',{message:'Update Q1 model sections 5.1-5.3 and triangular counterexample figure'}));
    const commit=await git(['rev-parse','HEAD']);
    const current={};
    for(const name of changedFiles) {
      const remote=await call('read_file',{filePath:name,mode:'full'});
      if(hash(remote)!==hash(payload[name])) throw new Error('Remote content verification failed: '+name);
      current[name]=hash(remote);
    }
    const tree=await git(['ls-tree','-r','HEAD']);
    const oldTree=(await readFile(path.join(evidence,'before_tree.txt'),'utf8')).trim();
    const otherEntries = t => t.split('\n').filter(line=>!changedFiles.includes(line.split('\t')[1])).sort().join('\n');
    if(otherEntries(tree)!==otherEntries(oldTree)) throw new Error('Other project files changed; inspect concurrent changes');
    const clean=!(await git(['status','--porcelain']));
    const branches=await git(['branch','-r','--contains','HEAD']);
    if(!clean||!branches.includes('origin/')) throw new Error('Push not confirmed by tracking branch');
    const report={projectId,commit,verifiedAt:new Date().toISOString(),files:current,otherFilesUnchanged:true,clean,pushed:true};
    await save('push_verified.json',JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify(report,null,2));
  } else throw new Error('Unknown mode');
} finally {await client.close();}
