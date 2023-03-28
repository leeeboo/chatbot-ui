import { ChatBody, Message, OpenAIModelID } from '@/types';
import { DEFAULT_SYSTEM_PROMPT } from '@/utils/app/const';
import { OpenAIStream } from '@/utils/server';
import tiktokenModel from '@dqbd/tiktoken/encoders/cl100k_base.json';
import { init, Tiktoken } from '@dqbd/tiktoken/lite/init';
// @ts-expect-error
import wasm from '../../node_modules/@dqbd/tiktoken/lite/tiktoken_bg.wasm?module';
import { PineconeClient } from "@pinecone-database/pinecone";

export const config = {
  runtime: 'edge',
};

const reset = "\x1b[0m";
const red = "\x1b[31m";
const green = "\x1b[32m";
const yellow = "\x1b[33m";
const blue = "\x1b[34m";
const magenta = "\x1b[35m";
const cyan = "\x1b[36m";

const handler = async (req: Request): Promise<Response> => {
  try {
    const { model, messages, key, prompt } = (await req.json()) as ChatBody;

    /////
    const userMessageContent = [...messages].reverse().find((message) => message.role === "user")?.content || "";
    
    const res = await fetch("https://api.openai.com/v1/embeddings", {
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${key ? key : process.env.OPENAI_API_KEY}`
      },
      method: "POST",
      body: JSON.stringify({
        model: "text-embedding-ada-002",
        input: userMessageContent,
      })
    });

    if (res.status !== 200) {
      const statusText = res.statusText; 
      throw new Error(`OpenAI API returned an error: ${statusText}`);
    }
    const embedding = await res.json();
    //console.log(embedding);

    const pinecone = new PineconeClient();
    await pinecone.init({
      environment: "us-central1-gcp",
      apiKey: process.env.PINECONE_API_KEY || '',
    });

    //pinecone.projectName = "chatwith"
    const index = pinecone.Index("shadow-tv");
    const queryRequest = {
      vector: embedding.data[0].embedding,
      topK: 3,
      includeValues: false,
      includeMetadata: true,
      namespace: "shadowtv20230326",
    };
    //console.log(queryRequest);
    let queryResponse;
    try {
      queryResponse = await index.query({ queryRequest });
    } catch (error) {
      console.error("Error querying Pinecone index:", error);
      queryResponse = null;
    }

    interface Metadata {
      text: string;
      bvid: string;
      ytid: string;
      title: string;
    }
    
    if (queryResponse && queryResponse.matches) {
      let others: string[] = [];
      const combinedTexts = queryResponse.matches
        .map((result) => {
          const metadata = result.metadata as Metadata; // Cast metadata to the Metadata interface
          return metadata?.text || '';
        })
        .filter((text) => text.trim().length > 0)
        .join(" ");

      console.log("用户问题:");
      console.log(`${yellow}${userMessageContent}${reset}`);

      for (const result of queryResponse.matches) {
        const metadata = result.metadata as Metadata; // Cast metadata to the Metadata interface
        
        const yt = `https://www.youtube.com/watch?v=${metadata.ytid}`;
        const bili = `https://www.bilibili.com/video/${metadata.bvid}`;

        const other = `相关视频: ${metadata.title}
        ${yt}`;
        others.push(other);
        // Process metadata as needed

        const metadataJsonString = JSON.stringify(metadata, null, 2);
        console.log("Metadata:");
        //console.log(metadata);

        console.log(`${green}${metadataJsonString}${reset}`);
      }
      //console.log("combinedTexts:", combinedTexts);
      const uniqueArrayOthers = Array.from(new Set(others));

      const template = `你扮演一个叫做"黑影儿"的人。根据段落中文字的内容回答问题。
不要自己创造答案，不要说除了答案以外的任何内容，如果无法确定答案就回答"抱歉！根据已有数据未查询到答案，您也可以尝试换一个方式提问"。如果段落中有错别字，请修改错别字后再使用。
答案请返回Markdown格式。必须将参考资料的内容拼接在答案之后。
问题: 
"""
${userMessageContent}
"""
段落:
"""
${combinedTexts}
"""
参考资料:
"""
${uniqueArrayOthers.join("\n")}
"""
第一人称答案 in Markdown:`;

      const lastUserMessageIndex = messages
        .map((message, index) => ({ message, index }))
        .filter(({ message }) => message.role === "user")
        .slice(-1)
        .map(({ index }) => index)[0];

      if (lastUserMessageIndex !== undefined) {
        messages[lastUserMessageIndex].content = template;
      }
    }

    await init((imports) => WebAssembly.instantiate(wasm, imports));
    const encoding = new Tiktoken(
      tiktokenModel.bpe_ranks,
      tiktokenModel.special_tokens,
      tiktokenModel.pat_str,
    );

    const tokenLimit = model.id === OpenAIModelID.GPT_4 ? 6000 : 3000;

    let promptToSend = prompt;
    if (!promptToSend) {
      promptToSend = DEFAULT_SYSTEM_PROMPT;
    }

    const prompt_tokens = encoding.encode(promptToSend);

    let tokenCount = prompt_tokens.length;
    let messagesToSend: Message[] = [];

    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i];
      const tokens = encoding.encode(message.content);

      if (tokenCount + tokens.length > tokenLimit) {
        break;
      }
      tokenCount += tokens.length;
      messagesToSend = [message, ...messagesToSend];
    }

    encoding.free();

    //console.log("messagesToSend:", messagesToSend);
    const stream = await OpenAIStream(model, promptToSend, key, messagesToSend);

    return new Response(stream);
  } catch (error) {
    console.error(error);
    return new Response('Error', { status: 500 });
  }
};

export default handler;
