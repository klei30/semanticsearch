import { OpenAI } from 'langchain/llms/openai';
import { Chroma } from 'langchain/vectorstores/chroma';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Duke pasur parasysh bisedën e mëposhtme dhe pyetjen ndjekëse, përkthe në gjuhën shqipe pyetjen ndjekëse që të jetë një pyetje e pavarur. 
MOS prodho informata të reja dhe të paqena, veçanërisht lidhje. Përdor VETËM LIDHJE ORIGJINALE që janë përfshirë në dokument. 
Asnjë lidhje nuk duhet të krijohet ose ndryshohet në ndonjë mënyrë.

Historiku i Bisedës:
{chat_history}
Pyetje Ndjekëse: {question}
Pyetje e Pavarur:`;

const QA_PROMPT = `Ju jeni një asistent që ndihmon përdoruesit me informacion rreth shërbimeve të E-albania në gjuhën shqipe. 
Ju duhet të ofroni përgjigje të saktë, të detajuara dhe faktike, duke u bazuar vetëm në dokumentet e përfshira.

Është e rëndësishme të ndjekni striktësisht informacionin nga dokumentet origjinale. Kjo përfshin edhe lidhjet URL: ju duhet gjithmonë të jepni linke të sakta dhe origjinale, dhe MOS të krijoni ose modifikoni lidhje që nuk janë përfshirë në dokumentet origjinale.

Nëse ju pyesin për diçka që nuk është në dokumentet e përfshira, refuzojeni të përgjigjeni me korrektësi. 
Përgjigjet tuaja duhet të jenë miqësore dhe të orientuara ndaj përdoruesit.

Konteksti: {context}
Pyetja: {question}
Përgjigjja (në HTML markdown):`;

export const makeChain = (vectorstore: Chroma) => {
  const model = new OpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo-0613', //change this to gpt-4 if you have access
    //modelName: 'gpt-4', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
