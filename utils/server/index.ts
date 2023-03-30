import { Message, OpenAIModel } from '@/types';
import {
  createParser,
  ParsedEvent,
  ReconnectInterval,
} from 'eventsource-parser';
import { OPENAI_API_HOST } from '../app/const';

let logMessages: string[] = [];
const reset = "\x1b[0m";
const red = "\x1b[31m";
const green = "\x1b[32m";
const yellow = "\x1b[33m";
const blue = "\x1b[34m";
const magenta = "\x1b[35m";
const cyan = "\x1b[36m";

export const OpenAIStream = async (
  model: OpenAIModel,
  systemPrompt: string,
  key: string,
  messages: Message[],
  question: string,
) => {
  const res = await fetch(`${OPENAI_API_HOST}/v1/chat/completions`, {
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${key ? key : process.env.OPENAI_API_KEY}`,
    },
    method: 'POST',
    body: JSON.stringify({
      model: model.id,
      messages: [
        {
          role: 'system',
          content: systemPrompt,
        },
        ...messages,
      ],
      max_tokens: 1000,
      temperature: 0.0,
      stream: true,
    }),
  });

  if (res.status !== 200) {
    const statusText = res.statusText;
    throw new Error(`OpenAI API returned an error: ${statusText}`);
  }

  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  const stream = new ReadableStream({
    async start(controller) {
      const onParse = (event: ParsedEvent | ReconnectInterval) => {
        if (event.type === 'event') {
          const data = event.data;

          if (data === '[DONE]') {
            console.log("GPT回答:");
            const answer = logMessages.join('');
            console.log(`${magenta}${answer}${reset}`);
            console.log("--------------------------------------------");
            
            fetch("https://open.feishu.cn/open-apis/bot/v2/hook/" + process.env.FEISHU_BOT, {
              headers: {
                "Content-Type": "application/json"
              },
              method: "POST",
              body: JSON.stringify({
                msg_type: "text",
                content: {
                  text: "问题: " + question + "\n答案: " + answer
                },
              })
            });
            logMessages = [];
            controller.close();
            return;
          }

          try {
            const json = JSON.parse(data);
            const text = json.choices[0].delta.content;
            
            logMessages.push(text);
            const queue = encoder.encode(text);
            controller.enqueue(queue);
          } catch (e) {
            controller.error(e);
          }
        }
      };

      const parser = createParser(onParse);

      for await (const chunk of res.body as any) {
        parser.feed(decoder.decode(chunk));
      }
    },
  });

  return stream;
};
