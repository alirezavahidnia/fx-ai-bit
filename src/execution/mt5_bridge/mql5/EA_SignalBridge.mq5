//+------------------------------------------------------------------+
//|                                                 EA_SignalBridge  |
//|            Minimal MT5 EA to receive JSON over TCP localhost     |
//+------------------------------------------------------------------+
#property strict
#include <Trade/Trade.mqh>
CTrade Trade;

#include <WinSock2.mqh>

int serverSocket = INVALID_SOCKET;
int clientSocket = INVALID_SOCKET;
ushort PORT = 18080;

string ReadLine(int sock)
{
   string result = "";
   char buffer[1024];
   int r;
   while((r = recv(sock, buffer, sizeof(buffer), 0)) > 0)
   {
      result += CharArrayToString(buffer, 0, r);
      int nl = StringFind(result, "\n");
      if(nl >= 0) break;
   }
   return result;
}

string JsonOk(string msg) { return "{\"ok\":true,\"msg\":\"" + msg + "\"}"; }
string JsonErr(string msg) { return "{\"ok\":false,\"error\":\"" + msg + "\"}"; }

string ExtractString(string src, string key, string def="")
{
   string tag = "\"" + key + "\":\"";
   int i = StringFind(src, tag);
   if(i < 0) return def;
   i += StringLen(tag);
   int j = StringFind(src, "\"", i);
   if(j < 0) return def;
   return StringSubstr(src, i, j - i);
}
double ExtractNumber(string src, string key, double def=0.0)
{
   string tag = "\"" + key + "\":";
   int i = StringFind(src, tag);
   if(i < 0) return def;
   i += StringLen(tag);
   int j = StringFind(src, ",", i);
   int k = StringFind(src, "}", i);
   if(j < 0 || (k >= 0 && k < j)) j = k;
   if(j < 0) j = i + 16;
   string s = StringTrim(StringSubstr(src, i, j - i));
   return (double)StringToDouble(s);
}

int OnInit()
{
   serverSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
   if(serverSocket == INVALID_SOCKET) return INIT_FAILED;

   sockaddr_in addr;
   addr.sin_family = AF_INET;
   addr.sin_port = htons(PORT);
   addr.sin_addr = inet_addr("127.0.0.1");
   if(bind(serverSocket, addr) != 0) return INIT_FAILED;
   if(listen(serverSocket, 1) != 0) return INIT_FAILED;
   Print("EA_SignalBridge listening on 127.0.0.1:", PORT);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   if(clientSocket != INVALID_SOCKET) closesocket(clientSocket);
   if(serverSocket != INVALID_SOCKET) closesocket(serverSocket);
}

void OnTick()
{
   int s = accept(serverSocket, NULL, NULL);
   if(s == INVALID_SOCKET) return;

   clientSocket = s;
   string req = ReadLine(clientSocket);
   if(StringLen(req) == 0)
   {
      send(clientSocket, JsonErr("empty request"));
      closesocket(clientSocket);
      clientSocket = INVALID_SOCKET;
      return;
   }

   if(StringFind(req, "\"cmd\":\"ping\"") >= 0)
   {
      send(clientSocket, "{\"event\":\"pong\"}");
   }
   else if(StringFind(req, "\"cmd\":\"close_all\"") >= 0)
   {
      for(int i = PositionsTotal() - 1; i >= 0; --i)
      {
         if(!PositionSelectByIndex(i)) continue;
         string sym = (string)PositionGetString(POSITION_SYMBOL);
         Trade.PositionClose(sym);
      }
      send(clientSocket, JsonOk("closed all"));
   }
   else if(StringFind(req, "\"cmd\":\"close_symbol\"") >= 0)
   {
      string sym = ExtractString(req, "symbol", _Symbol);
      bool any = false;
      for(int i = PositionsTotal() - 1; i >= 0; --i)
      {
         if(!PositionSelectByIndex(i)) continue;
         string ps = (string)PositionGetString(POSITION_SYMBOL);
         if(ps == sym) { any = true; Trade.PositionClose(ps); }
      }
      send(clientSocket, any ? JsonOk("closed "+sym) : JsonErr("no positions for "+sym));
   }
   else if(StringFind(req, "\"cmd\":\"place_order\"") >= 0)
   {
      string sym = ExtractString(req, "symbol", _Symbol);
      string side = ExtractString(req, "side", "buy");
      double vol = ExtractNumber(req, "volume", 0.10);
      double sl = ExtractNumber(req, "sl", 0.0);
      double tp = ExtractNumber(req, "tp", 0.0);

      bool ok = false;
      if(side == "buy")  ok = Trade.Buy(vol, sym, 0.0, sl, tp);
      if(side == "sell") ok = Trade.Sell(vol, sym, 0.0, sl, tp);

      send(clientSocket, ok ? JsonOk("placed") : JsonErr("trade failed"));
   }
   else
   {
      send(clientSocket, JsonErr("unknown cmd"));
   }

   closesocket(clientSocket);
   clientSocket = INVALID_SOCKET;
}
