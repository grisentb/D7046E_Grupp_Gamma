// fetch data
debug = false;

async function requestData() {
     $.ajax({
        type: 'GET',
        url: '/data',
        success :(data) => {
            updateData(data);
            if(debug)
                $('#debug').text(data)
        }
    })
        .then(function () {
        setTimeout(requestData, 50) //call itself every 50ms
    });
}
// async function requestStructureData() {
//       $.ajax({
//         type: 'GET',
//         url: '/data/structures',
//         success :(data) => {
//             updateStructureData(data);
//             if(debug)
//                 $('#debug').text(data)
//         }
//     });
// }
//
// async function requestActivity() {
//       $.ajax({
//         type: 'GET',
//         url: '/data/events',
//         success :(data) => {
//             updateActivityData(data);
//             if(debug)
//                 $('#debug').text(data)
//         }
//     }).then(function () {
//           setTimeout(requestActivity,500)
//       });
// }
requestData();
// requestStructureData();
// requestActivity();
