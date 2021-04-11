"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.deactivate = exports.activate = void 0;
// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
const axios_1 = require("axios");
const vscode = require("vscode");
// this method is called when your extension is activated
// your extension is activated the very first time the command is executed
function activate(context) {
    // Use the console to output diagnostic information (console.log) and errors (console.error)
    // This line of code will only be executed once when your extension is activated
    console.log('Congratulations, your extension "codedecoder" is now active!');
    // The command has been defined in the package.json file
    // Now provide the implementation of the command with registerCommand
    // The commandId parameter must match the command field in package.json
    let disposable = vscode.commands.registerCommand('codedecoder.helloWorld', () => {
        // The code you place here will be executed every time your command is executed
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }
        const document = editor.document;
        const selection = editor.selection;
        let code = document.getText(selection);
        const headers = {
            // eslint-disable-next-line @typescript-eslint/naming-convention
            'Content-Type': 'application/json'
        };
        axios_1.default.post("http://b631c1d73e70.ngrok.io", { code: code }, {
            headers: headers
        }).then(function (response) {
            if (editor) {
                editor.edit(editBuilder => {
                    editBuilder.insert(selection.start, "# " + response.data + "\n");
                });
            }
            vscode.window.showInformationMessage(response.data);
        });
        // Display a message box to the user
        // vscode.window.showInformationMessage(code);
    });
    context.subscriptions.push(disposable);
}
exports.activate = activate;
// this method is called when your extension is deactivated
function deactivate() { }
exports.deactivate = deactivate;
//# sourceMappingURL=extension.js.map