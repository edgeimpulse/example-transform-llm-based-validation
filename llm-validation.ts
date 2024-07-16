import fs from 'fs';
import Path from 'path';
import program from 'commander';
import { EdgeImpulseApi } from 'edge-impulse-api';
import * as models from 'edge-impulse-api/build/library/sdk/model/models';
import OpenAI from "openai";
import asyncPool from 'tiny-async-pool';

const packageVersion = (<{ version: string }>JSON.parse(fs.readFileSync(
    Path.join(__dirname, '..', 'package.json'), 'utf-8'))).version;

if (!process.env.EI_PROJECT_API_KEY) {
    console.log('Missing EI_PROJECT_API_KEY');
    process.exit(1);
}
if (!process.env.OPENAI_API_KEY) {
    console.log('Missing OPENAI_API_KEY');
    process.exit(1);
}

let API_URL = process.env.EI_API_ENDPOINT || 'https://studio.edgeimpulse.com/v1';
const API_KEY = process.env.EI_PROJECT_API_KEY;

API_URL = API_URL.replace('/v1', '');

program
    .description('Label using an LLM ' + packageVersion)
    .version(packageVersion)

    .requiredOption('--validation1 <prompt>',
            `Disable the sample if this prompt is true` +
            `E.g. "The bounding boxes and labels do not correspond to to the objects in the image" `)
    .requiredOption('--validation2 <prompt>',
        `Disable the sample if this prompt is true` +
        `E.g. "The bounding boxes and labels do not correspond to to the objects in the image" `)
    .requiredOption('--validation3 <prompt>',
            `Disable the sample if this prompt is true` +
            `E.g. "The bounding boxes and labels do not correspond to to the objects in the image" `)
    .option('--limit <n>', `Max number of samples to process`)
    .option('--concurrency <n>', `Concurrency (default: 1)`)
    .option('--verbose', 'Enable debug logs')
    .allowUnknownOption(true)
    .parse(process.argv);

const api = new EdgeImpulseApi({ endpoint: API_URL });

const validation1Argv = <string>program.validation1;
const validation2Argv = <string>program.validation2;
const validation3Argv = <string>program.validation3;
const disableLabelsArgv = ["invalid"]
const limitArgv = program.limit ? Number(program.limit) : undefined;
const concurrencyArgv = program.concurrency ? Number(program.concurrency) : 1;


// eslint-disable-next-line @typescript-eslint/no-floating-promises
(async () => {
    try {
        const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

        await api.authenticate({
            method: 'apiKey',
            apiKey: API_KEY,
        });

        // listProjects returns a single project if authenticated by API key
        const project = (await api.projects.listProjects()).projects[0];

        console.log(`Validating data for "${project.owner} / ${project.name}"`);
        console.log(`    Validation1: "${validation1Argv}"`);
        console.log(`    Validation2: "${validation2Argv}"`);
        console.log(`    Validation3: "${validation3Argv}"`);
        console.log(`    Limit no. of samples to label to: ${typeof limitArgv === 'number' ? limitArgv.toLocaleString() : 'No limit'}`);
        console.log(`    Concurrency: ${concurrencyArgv}`);
      
        console.log(`Finding enabled data...`);
        const { enabledSamples, uniqueLabels } = await listEnabledData(project.id);
        //find list of unique labels in enabledSamples.s.boundingBoxes

        console.log(`Unique labels in enabled data: ${Array.from(uniqueLabels).join(', ')}`);

        console.log(`Finding enabled data OK (found ${enabledSamples.length} samples)`);
        console.log(``);

        const total = typeof limitArgv === 'number' ?
            (enabledSamples.length > limitArgv ? limitArgv : enabledSamples.length) :
            enabledSamples.length;
        let processed = 0;
        let error = 0;
        let labelCount: { [k: string]: number } = { };

        const getSummary = () => {
            let labelStr = Object.keys(labelCount).map(k => k + '=' + labelCount[k]).join(', ');
            if (labelStr.length > 0) {
                return `(${labelStr}, error=${error})`;
            }
            else {
                return `(error=${error})`;
            }
        };

        let updateIv = setInterval(async () => {
            let currFile = (processed).toString().padStart(total.toString().length, ' ');
            console.log(`[${currFile}/${total}] Labeling samples... ` +
                getSummary());
        }, 3000);

        const labelSampleWithOpenAI = async (sample: models.Sample) => {
            try {
                const formattedBoundingBoxes = sample.boundingBoxes.map((box: { label: string; x: any; y: any; width: any; height: any; }) => {
                    return `Label: ${box.label.trim()}, Location: (x: ${box.x}, y: ${box.y}), Size: (width: ${box.width}, height: ${box.height})`;
                  }).join('\n');
                const json = await retryWithTimeout(async () => {
                    const imgBuffer = await api.rawData.getSampleAsImage(project.id, sample.id, { });
                    
                    const resp = await openai.chat.completions.create({
                        model: 'gpt-4o-2024-05-13',
                        messages: [{
                        role: 'system',
                        content: `You always respond with the following JSON structure, regardless of the prompt: \`{ "label": "XXX", "reason": "YYY" }\`. ` +
                                `Put the requested answer in 'label', and put your detailed reasoning in 'reason' including the name of the objects if mentioned.`
                        }, {
                            role: 'user',
                            content: [{
                                type: 'text',
                                text: `The following image is from a dataset with these possible labels: \`${Array.from(uniqueLabels).join(', ')}\`. ` +
                                `This image has labeled bounding boxes consisting of: \`${formattedBoundingBoxes}\`. ` +
                                `Respond with \"invalid\" if:
                                - \`${validation1Argv}\`
                                - \`${validation2Argv}\`
                                - \`${validation3Argv}\`
                                
                                Otherwise respond with \"valid\".`,
                            }, {
                                type: 'image_url',
                                image_url: {
                                    url: 'data:image/jpeg;base64,' + (imgBuffer.toString('base64')),
                                    detail: 'auto'
                                }
                            }]
                        }]
                    });

                    console.log('resp', JSON.stringify(resp, null, 4));

                    if (resp.choices.length !== 1) {
                        throw new Error('Expected choices to have 1 item (' + JSON.stringify(resp) + ')');
                    }
                    if (resp.choices[0].message.role !== 'assistant') {
                        throw new Error('Expected choices[0].message.role to equal "assistant" (' + JSON.stringify(resp) + ')');
                    }
                    if (typeof resp.choices[0].message.content !== 'string') {
                        throw new Error('Expected choices[0].message.content to be a string (' + JSON.stringify(resp) + ')');
                    }

                    let jsonContent: { label: string, reason: string };
                    try {
                        jsonContent = <{ label: string, reason: string }>JSON.parse(resp.choices[0].message.content);
                        if (typeof jsonContent.label !== 'string') {
                            throw new Error('label was not of type string');
                        }
                        if (typeof jsonContent.reason !== 'string') {
                            throw new Error('reason was not of type string');
                        }
                    }
                    catch (ex2) {
                        let ex = <Error>ex2;
                        throw new Error('Failed to parse message content: ' + (ex.message + ex.toString()) +
                            ' (raw string: "' + resp.choices[0].message.content + '")');
                    }

                    return jsonContent;
                }, {
                    fnName: 'completions.create',
                    maxRetries: 3,
                    onWarning: (retriesLeft, ex) => {
                        let currFile = (processed).toString().padStart(total.toString().length, ' ');
                        console.log(`[${currFile}/${total}] WARN: Failed to label ${sample.filename} (ID: ${sample.id}): ${ex.message || ex.toString()}. Retries left=${retriesLeft}`);
                    },
                    onError: (ex) => {
                        let currFile = (processed).toString().padStart(total.toString().length, ' ');
                        console.log(`[${currFile}/${total}] ERR: Failed to label ${sample.filename} (ID: ${sample.id}): ${ex.message || ex.toString()}.`);
                    },
                    timeoutMs: 60000,
                });

                await retryWithTimeout(async () => {
                    if (disableLabelsArgv.indexOf(json.label) > -1) {
                        await api.rawData.disableSample(project.id, sample.id);
                    }
                    // update metadata
                    sample.metadata = sample.metadata || {};
                    sample.metadata.reason = json.reason;
                    sample.metadata.validation = json.label
                    sample.metadata.formattedBoundingBoxes = formattedBoundingBoxes;
                    await api.rawData.setSampleMetadata(project.id, sample.id, {
                        metadata: sample.metadata,
                    });
                }, {
                    fnName: 'edgeimpulse.api',
                    maxRetries: 3,
                    timeoutMs: 60000,
                    onWarning: (retriesLeft, ex) => {
                        let currFile = (processed).toString().padStart(total.toString().length, ' ');
                        console.log(`[${currFile}/${total}] WARN: Failed to update metadata for ${sample.filename} (ID: ${sample.id}): ${ex.message || ex.toString()}. Retries left=${retriesLeft}`);
                    },
                    onError: (ex) => {
                        let currFile = (processed).toString().padStart(total.toString().length, ' ');
                        console.log(`[${currFile}/${total}] ERR: Failed to update metadata for ${sample.filename} (ID: ${sample.id}): ${ex.message || ex.toString()}.`);
                    },
                });

                if (!labelCount[json.label]) {
                    labelCount[json.label] = 0;
                }
                labelCount[json.label]++;
            }
            catch (ex2) {
                let ex = <Error>ex2;
                let currFile = (processed + 1).toString().padStart(total.toString().length, ' ');
                console.log(`[${currFile}/${total}] Failed to validate sample "${sample.filename}" (ID: ${sample.id}): ` +
                    (ex.message || ex.toString()));
                error++;
            }
            finally {
                processed++;
            }
        };

        try {
            console.log(`validating ${total.toLocaleString()} samples...`);

            await asyncPool(concurrencyArgv, enabledSamples.slice(0, total), labelSampleWithOpenAI);

            clearInterval(updateIv);

            console.log(`[${total}/${total}] Validating samples... ` + getSummary());
            console.log(`Done validating samples, goodbye!`);
        }
        finally {
            clearInterval(updateIv);
        }
    }
    catch (ex2) {
        let ex = <Error>ex2;
        console.log('Failed to validate data:', ex.message || ex.toString());
        process.exit(1);
    }

    process.exit(0);
})();

async function listEnabledData(projectId: number) {
    const limit = 1000;
    let offset = 0;
    let allSamples: models.Sample[] = [];
    let uniqueLabels = new Set<string>();

    let iv = setInterval(() => {
        console.log(`Still finding enabled data (found ${allSamples.length} samples)...`);
    }, 3000);

    try {
        while (1) {
            let ret = await api.rawData.listSamples(projectId, {
                category: 'training',
                offset: offset,
                limit: limit,
            });
            
            if (ret.samples.length === 0) {
                break;
            }
            for (let s of ret.samples) {
               
                
                if (s.chartType === 'image' && s.boundingBoxes.length > 0) {
                     // find unique labels
                    for (let label of s.boundingBoxes) {
                        uniqueLabels.add(label["label"]);
                    }
                    if (s.isDisabled === false) {   
                        allSamples.push(s);
                    }
                }
            }
            offset += limit;
        }
        while (1) {
            let ret = await api.rawData.listSamples(projectId, {
                category: 'testing',
                offset: offset,
                limit: limit,
            });
            if (ret.samples.length === 0) {
                break;
            }
            for (let s of ret.samples) {
                if (s.chartType === 'image' && s.boundingBoxes.length > 0) {
                    // find unique labels
                   for (let label of s.boundingBoxes) {
                        uniqueLabels.add(label["label"]);
                }

                   if (s.isDisabled === false) {   
                       allSamples.push(s);
                   }
               }
            }
            offset += limit;
        }
    }
    finally {
        clearInterval(iv);
    }
    return { enabledSamples: allSamples, uniqueLabels: uniqueLabels };
}



export async function retryWithTimeout<T>(fn: () => Promise<T>, opts: {
    fnName: string,
    timeoutMs: number,
    maxRetries: number,
    onWarning: (retriesLeft: number, ex: Error) => void,
    onError: (ex: Error) => void,
}) {
    const { timeoutMs, maxRetries, onWarning, onError } = opts;

    let retriesLeft = maxRetries;

    let ret: T;

    while (1) {
        try {
            ret = await new Promise<T>(async (resolve, reject) => {
                let timeout = setTimeout(() => {
                    reject(opts.fnName + ' did not return within ' + timeoutMs + 'ms.');
                }, timeoutMs);

                try {
                    const b = await fn();

                    resolve(b);
                }
                catch (ex) {
                    reject(ex);
                }
                finally {
                    clearTimeout(timeout);
                }
            });

            break;
        }
        catch (ex2) {
            let ex = <Error>ex2;

            retriesLeft = retriesLeft - 1;
            if (retriesLeft === 0) {
                onError(ex);
                throw ex2;
            }

            onWarning(retriesLeft, ex);
        }
    }

    return ret!;
}
